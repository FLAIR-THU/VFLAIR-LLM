import sys, os
sys.path.append(os.pardir)

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import re
import time
import numpy as np
import pickle
import json
import collections
from loguru import logger
from torch.distributions.laplace import Laplace
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteriaList,StoppingCriteria


from party.party import Party
from party.llm_party import Party as Party_LLM

from dataset.party_dataset import *
from dataset.party_dataset import ActiveDataset
from evaluates.defenses.defense_api import apply_defense

from load.LoadModels import load_models_per_party, QuestionAnsweringModelOutput
from utils import timer
from utils.squad_utils import normalize_answer, _get_best_indexes, compute_exact, compute_f1
from utils.communication_protocol_funcs import get_size_of
from utils.communication_protocol_funcs import compress_pred
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor
from utils.cluster_utils import run_cluster, redivide_cluster, CenterLoss, load_cluster_results, save_cluster_results
from utils.inferdpt_utils import *
from utils.custext_utils import *
from utils.snd_utils import *

from models.denoise_model import *
from models.imagined_adversary_models import *
from models.adversarial_model import *
from models.mid_model_rapper import *
from models.llm_models.base import VFLModelIntermediate

from config import vfl_basic_config

from tqdm import tqdm


DATASET_TYPE_DICT = {
    'GMS8K': GSMDataset_LLM
}
class PassiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)

    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        # self.train_dst = TensorDataset(train_inputs, train_masks) # the second label is just a place holder
        # self.test_dst = TensorDataset(test_inputs, test_masks) # the second label is just a place holder

        self.train_dst = PassiveDataset(self.train_data)
        self.test_dst = PassiveDataset(self.test_data)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)


class PassiveParty_LLM(Party_LLM):
    def __init__(self, args, index, need_data=True, need_model=True):
        print(f'###### initialize PassiveParty_LLM : Party {index} ######')
        super().__init__(args, index, need_data=need_data, need_model=need_model)
        logger.debug(f'running on cuda{os.getenv("CUDA_VISIBLE_DEVICES").split(",")[torch.cuda.current_device()]}')

        self.init_apply_defense(args.apply_defense, args.apply_adversarial, 
                                args.defense_configs, args.main_lr,
                                args.device)

        self.criterion = cross_entropy_for_onehot

        # self.encoder = args.encoder
        self.train_index = None  # args.idx_train
        self.test_index = None  # args.idx_test

        self.device = args.device

        self.gt_one_hot_label = None
        self.clean_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])

        self.global_pred = None
        self.global_loss = None
        self.communication_cost = 0
        self.num_total_comms = 0
        self.current_step = 0

        self.num_labels = args.num_classes
        self.weights_grad_a = None  # no gradient for model in passive party(no model update)
        self.weights_grad_a_tail = None
        # self.encoder_trainable = args.encoder_trainable[index]

    def init_apply_defense(self, need_apply_defense, apply_adversarial, defense_configs, main_lr, device):
        # some defense need model, add here
        if need_apply_defense:
            if apply_adversarial and (self.index in defense_configs["party"]):
                print(f'Passive Party {self.index}: init Adversarial Trainining Defense -- {self.args.defense_param}')
                
                if not 'party' in defense_configs:
                    defense_configs['party'] = [0]
                    print('[warning] default passive party selected for applying adversarial training')
                # read adversarial defense configs
                self.ad_position = defense_configs['position']
                self.adversarial_model_lr = defense_configs['adversarial_model_lr']
                self.adversarial_model_hidden_size = defense_configs['adversarial_model_hidden_size'] if (
                        'adversarial_model_hidden_size' in defense_configs) else 80
                if not ('adversarial_model' in defense_configs):
                    adversarial_model_name = 'Adversarial_Mapping'
                else:
                    adversarial_model_name = defense_configs['adversarial_model']
                seq_length = defense_configs['seq_length']
                embed_dim = self.args.model_embedded_dim  # defense_configs['embed_dim']

                if self.args.model_architect=='CLS':
                    self.label_size = self.args.num_classes
                imagined_adversary_model_name = defense_configs['imagined_adversary']
                tail_imagined_adversary_model_name = defense_configs['tail_imagined_adversary']  if (
                        'tail_imagined_adversary' in defense_configs) else 'ImaginedAdversary_Tail_MLP3'

                self.imagined_adversary_hidden_size = defense_configs['imagined_adversary_hidden_size'] if (
                        'imagined_adversary_hidden_size' in defense_configs) else 80
                self.imagined_adversary_lr = defense_configs['imagined_adversary_lr']

                self.adversary_crit = nn.CrossEntropyLoss()
                self.adversary_lambda = defense_configs['lambda']

                # AD at model head
                if 'head' in self.ad_position:
                    # prepare adversarial model --  for adversarial training
                    self.head_adversarial_model = globals()[adversarial_model_name](seq_length, embed_dim,
                                                    self.adversarial_model_hidden_size).to(self.args.device)
                    if self.local_model_optimizer == None:
                        self.local_model_optimizer = torch.optim.Adam(self.head_adversarial_model.parameters(),
                                                                    lr=self.adversarial_model_lr)
                    else:
                        self.local_model_optimizer.add_param_group(
                            {'params': self.head_adversarial_model.parameters(), 'lr': self.adversarial_model_lr})

                    # prepare imagined adversary --  for adversarial training
                    self.head_imagined_adversary = globals()[imagined_adversary_model_name](seq_length, embed_dim,
                                                            self.imagined_adversary_hidden_size).to(device)

                    self.head_imagined_adversary_optimizer = torch.optim.Adam(list(self.head_imagined_adversary.parameters()),
                                                                        lr=self.imagined_adversary_lr)
                # AD at model tail
                if 'tail' in self.ad_position:
                    # prepare adversarial model --  for adversarial training
                    self.tail_adversarial_model = globals()[adversarial_model_name](seq_length, embed_dim,
                                                    self.adversarial_model_hidden_size).to(self.args.device)
                    if self.local_model_tail_optimizer == None:
                        self.local_model_tail_optimizer = torch.optim.Adam(self.tail_adversarial_model.parameters(),
                                                                    lr=self.adversarial_model_lr)
                    else:
                        self.local_model_tail_optimizer.add_param_group(
                            {'params': self.tail_adversarial_model.parameters(), 'lr': self.adversarial_model_lr})

                    # prepare imagined adversary --  for adversarial training
                    self.tail_imagined_adversary = globals()[tail_imagined_adversary_model_name](seq_length, embed_dim,self.label_size,
                                                    self.adversarial_model_hidden_size).to(self.args.device)
                    self.tail_imagined_adversary_optimizer = torch.optim.Adam(list(self.tail_imagined_adversary.parameters()),
                                                                    lr=self.imagined_adversary_lr)

            elif self.args.apply_mid and (self.index in self.args.defense_configs["party"]):
                print(f'Passive Party {self.index}: init MID Defense')
                self.mid_lambda = self.args.defense_configs['lambda']
                self.mid_model_name = self.args.defense_configs['mid_model_name']
                self.mid_lr = self.args.defense_configs['lr']
                self.squeeze_dim = self.args.defense_configs[
                    'squeeze_dim'] if 'squeeze_dim' in self.args.defense_configs else 0

                self.mid_position = self.args.defense_configs['mid_position'] \
                    if 'mid_position' in self.args.defense_configs else "head"  # "inner"

                current_bottleneck_scale = int(self.args.defense_configs['bottleneck_scale']) \
                    if 'bottleneck_scale' in self.args.defense_configs else 1

                if 'std_shift_hyperparameter' in self.args.defense_configs:
                    std_shift_hyperparameter = int(self.args.defense_configs['std_shift_hyperparameter'])
                else:
                    std_shift_hyperparameter = 5

                model_dtype = self.args.model_dtype
                print('self.args.model_dtype:',self.args.model_dtype)
                seq_length = self.args.defense_configs['seq_length']
                embed_dim = self.args.model_embedded_dim  # defense_configs['embed_dim']
                label_size = self.args.num_classes
                mid_model_path = self.args.defense_configs[
                    'mid_model_path'] if 'mid_model_path' in self.args.defense_configs else None

                print('init defense: mid on model head')
                print(self.mid_model_name)

                if mid_model_path != None and mid_model_path != "":
                    print('Load Mid Model:', mid_model_path)
                    with open(mid_model_path, 'rb') as f:
                        self.mid_model = pickle.load(f)
                else:
                    self.mid_model = globals()[self.mid_model_name](seq_length, embed_dim, label_size,\
                                                                    mid_lambda=self.mid_lambda,
                                                                    bottleneck_scale=current_bottleneck_scale,
                                                                    std_shift=std_shift_hyperparameter,
                                                                    model_dtype=model_dtype).to(
                        self.args.device)

                if self.mid_position == "head" or "head" in self.mid_position:
                    # if mid_model_path != None and mid_model_path != "":
                    #     print('Load Mid Model on head:', mid_model_path)
                    #     with open(mid_model_path, 'rb') as f:
                    #         self.mid_model = pickle.load(f)
                    # else:
                    self.head_mid_model = globals()[self.mid_model_name](seq_length, embed_dim, label_size,\
                                                                    mid_lambda=self.mid_lambda,
                                                                    bottleneck_scale=current_bottleneck_scale,
                                                                    std_shift=std_shift_hyperparameter,
                                                                    model_dtype=model_dtype).to(
                        self.args.device)

                    if self.local_model_optimizer == None:
                        self.local_model_optimizer = torch.optim.Adam(self.head_mid_model.parameters(), lr=self.mid_lr)
                    else:
                        self.local_model_optimizer.add_param_group(
                            {'params': self.head_mid_model.parameters(), 'lr': self.mid_lr})
                
                if self.mid_position == "tail" or "tail" in self.mid_position:
                    self.tail_mid_model = globals()[self.mid_model_name](seq_length, embed_dim, label_size,\
                                                                    mid_lambda=self.mid_lambda,
                                                                    bottleneck_scale=current_bottleneck_scale,
                                                                    std_shift=std_shift_hyperparameter,
                                                                    model_dtype=model_dtype).to(
                        self.args.device)

                    print('mid_position:', self.mid_position)
                    if self.local_model_tail_optimizer == None:
                        self.local_model_tail_optimizer = torch.optim.Adam(self.tail_mid_model.parameters(), lr=self.mid_lr)
                    else:
                        self.local_model_tail_optimizer.add_param_group(
                            {'params': self.tail_mid_model.parameters(), 'lr': self.mid_lr})
                print(f'self.mid_model_name:{self.mid_model_name}')

            elif self.args.apply_dp and (self.index in defense_configs["party"]):
                print(f'Passive Party {self.index}: init DP Defense')
            
            elif self.args.apply_textobfuscator and (self.index in defense_configs["party"]):
                print(f'Passive Party {self.index}: init TextObfuscate Defense')
                ## Clustering
                clustering_file_path = f'./models/model_parameters/clustering_models/{self.args.dataset}/cluster_num_{self.args.cluster_num}/'
                if not os.path.exists(clustering_file_path):
                    os.makedirs(clustering_file_path)
                if os.path.exists(clustering_file_path+'/cluster_center.pt') and os.path.exists(clustering_file_path+'/token2cluster.json'):
                    print(f'load cluster model from {clustering_file_path}')
                    self.token2cluster, self.cluster_center = load_cluster_results(clustering_file_path)
                else:
                    print(f'do clustering, save cluster model into {clustering_file_path}')
                    self.token2cluster, self.cluster_center = run_cluster(\
                        self.local_model, self.train_loader, cluster_num = self.args.cluster_num, cluster_method=self.args.cluster_method)
                    save_cluster_results(token2cluster=self.token2cluster, cluster_center=self.cluster_center, \
                        data_dir=clustering_file_path)
                
                ## Init Privacy Loss
                self.obfuscator_privacy_loss_func = CenterLoss(cluster_num=self.args.cluster_num, hidden_size = self.args.model_embedded_dim,\
                    w_cluster_close = self.args.w_cluster_close , w_cluster_away=self.args.w_cluster_away).to(self.local_model.device)
                self.obfuscator_privacy_loss_func.cluster_embedding.weight.data = self.cluster_center.type_as(self.obfuscator_privacy_loss_func.cluster_embedding.weight.data)
                
                self.obfuscator = Laplace(loc=torch.tensor(0, device=self.local_model.device, dtype=float), \
                    scale=torch.tensor(1/self.args.epsilon, device=self.local_model.device, dtype=float))
                
            elif self.args.apply_inferdpt and (self.index in defense_configs["party"]):
                print(f'Passive Party {self.index}: init InferDPT Defense')
                infer_dpt_kit_dir = f'./models/model_parameters/inferdpt_kit/{self.args.dataset}/'
                if not os.path.exists(infer_dpt_kit_dir):
                    os.makedirs(infer_dpt_kit_dir)
                
                # Init InferDPT Kit: token_to_vector_dict / sorted_similarities / delta_f
                if not os.path.exists(infer_dpt_kit_dir + '/token_2_vector.json'):
                    #### Generate InferDPT
                    embeddings = self.args.tokenizer.get_vocab()
                    print('embeddings:',type(embeddings),len(embeddings))
                    dtype = np.float32
                    embedding_weights = self.local_model.get_input_embeddings().weight
                    embedding_weights_np = embedding_weights.detach().cpu().numpy().astype(dtype)
                    filtered_index2token = filter_tokens(embeddings)
                    used_num_tokens = len(filtered_index2token)
                    print('used_num_tokens:',used_num_tokens)
                    
                    token_2_embedding = {}
                    for idx, token in filtered_index2token.items():
                        token_2_embedding[token] = embedding_weights_np[idx].tolist()
                    token_list = list(token_2_embedding.keys())
                    embedding_matrix = np.array(list(token_2_embedding.values()), dtype=dtype)
                    print('token_list:',len(token_list))
                    print('embedding_matrix:',embedding_matrix.shape)
                    self.token_to_vector_dict, self.sorted_similarities, self.delta_f = generate_inferdpt_kit(embedding_matrix,token_list)

                    # Save InferDPT kit
                    with open(infer_dpt_kit_dir+'/token_2_vector.json', 'w', encoding='utf8') as f:
                        json.dump(self.token_to_vector_dict, f, ensure_ascii=False, cls=NumpyEncoder)
                    with open(infer_dpt_kit_dir+'/sorted_similarities.json', 'w') as f:
                        json.dump(self.sorted_similarities, f, cls=NumpyEncoder)
                    with open(infer_dpt_kit_dir+'/delta_f.json', 'w') as f:
                        json.dump(self.delta_f, f, cls=NumpyEncoder)
                    print(f'Save InferDPTkit into:',infer_dpt_kit_dir)
                else:
                    print(f'Load InferDPTkit from:',infer_dpt_kit_dir)
                    with open(infer_dpt_kit_dir+'/token_2_vector.json', 'r', encoding='utf8') as f:
                        self.token_to_vector_dict = json.load(f)
                    with open(infer_dpt_kit_dir+'/sorted_similarities.json', 'r') as f:
                        self.sorted_similarities = json.load(f)
                    with open(infer_dpt_kit_dir+'/delta_f.json', 'r') as f:
                        self.delta_f = np.array(json.load(f))
                       
                if self.args.decode_model_path != "":
                    
                    self.decode_model = AutoModelForCausalLM.from_pretrained(self.args.decode_model_path,**self.args.decode_model_load_kwargs)
                    self.decode_model_tokenizer = AutoTokenizer.from_pretrained(self.args.decode_model_path)
                    # if self.args.dataset in ['GMS8K-test']:
                    #     self.decode_template = """The "Perturbed Answer" is answer to the "Original Math Problrm". Your task is to extract coherent and consistent answer from the "Perturbed Answer" to make it seamlessly align with the context established by the "Original Math Problem". Provide only your "Extracted Answer"\n\n——"Original Math Problem":{prefix}\n\n——"Perturbed Answer":{perturbed_answer}\n\n——"Extracted Answer":"""
                    # else:
                    self.decode_template = """Your task is to extend the "Prefix Text". Use the "Perturbed Generation" as your primary writing material for your extension. Extract coherent and consistent text from the "Perturbed Generation" and integrate them into your continuation. Ensure a seamless alignment with the context established by the "Prefix Text". Provide only your "Extended Text"\n——"Prefix Text":{prefix}\n——"Perturbed Generation":{perturbed_answer}\n——"Extended Text":"""

            elif self.args.apply_snd and (self.index in defense_configs["party"]):
                print(f'Passive Party {self.index}: init Split and Denoise Defense')
                #### Initialize denoise model
                embed_dim = self.args.model_embedded_dim
                denoise_model_type_dict = {
                    'denoiseModelv3': denoiseModelv3,
                    'denoiseModelv3_2slice': denoiseModelv3_2slice,
                    'denoiseModelv3_3slice': denoiseModelv3_3slice
                }
                assert self.args.denoise_model in denoise_model_type_dict.keys(), f"{denoise_model_type_dict} not supported"
                if self.args.vfl_model_slice_num == 3:
                    d_out = embed_dim
                else:
                    if self.args.model_architect == 'CLS':
                        d_out = self.args.num_classes
                    elif self.args.model_architect == 'CLM':
                        d_out = self.args.config.vocab_size
                    else:
                        assert 1>2, f"{self.args.model_architect} not supported for SnD defense"
                self.denoise_mod = denoise_model_type_dict[self.args.denoise_model]\
                    (d_model=embed_dim, d_out=d_out, args=self.args).to(self.args.device)
                
                denoise_model_dir = f"./models/model_parameters/denoise_model/{self.args.vfl_model_slice_num}-slice/{self.args.dataset}/"
                if not os.path.exists(denoise_model_dir):
                    os.makedirs(denoise_model_dir)
                self.denoise_model_path = denoise_model_dir + f"/{self.args.denoise_model}_eta{str(self.args.train_eta)}"
                
                if os.path.exists(self.denoise_model_path):
                    print('Load Denoise Model From: ',self.denoise_model_path)
                    self.denoise_mod.load_state_dict(torch.load(self.denoise_model_path, map_location=self.args.device))
                else:
                    print('Denosie Model Unprepared, will be trained later afetr party initialization')
                    
            elif self.args.apply_custext and (self.index in defense_configs["party"]):
                print(f'Passive Party {self.index}: init CUSTEXT Defense')
                custext_dict_path = f"./models/model_parameters/custext_kit/{self.args.dataset}/eps_{self.args.epsilon}_top_{self.args.topk}/"
                if not os.path.exists(custext_dict_path):
                    os.makedirs(custext_dict_path)
                
                if os.path.exists(custext_dict_path+'/p_dict.txt') and os.path.exists(custext_dict_path+'/sim_word_dict.txt'):
                    print(f'load custext p_dict/sim_word_dict from {custext_dict_path}')
                    with open(custext_dict_path+f"/p_dict.txt", 'r') as dic:
                        self.p_dict = json.load(dic)
                    with open(custext_dict_path+f"/sim_word_dict.txt", 'r') as dic:
                        self.sim_word_dict = json.load(dic)
                else:
                    self.p_dict, self.sim_word_dict = get_customized_mapping(self.train_data, defense_configs['epsilon'],defense_configs['topk'])
                    with open(custext_dict_path+f"/p_dict.txt", 'w') as json_file:
                        json_file.write(json.dumps(self.p_dict, ensure_ascii=False, indent=4))
                    with open(custext_dict_path+f"/sim_word_dict.txt", 'w') as json_file:
                        json_file.write(json.dumps(self.sim_word_dict, ensure_ascii=False, indent=4))
                
                
    def prepare_data(self, args, index):
        if not args.dataset:
            return None
        super().prepare_data(args, index)  # Party_llm's prepare_data

        if args.dataset in ['Alpaca', 'CodeAlpaca']:
            self.train_dst = AlpacaDataset_LLM(args, self.train_data, self.train_label, 'train')
            self.test_dst = AlpacaDataset_LLM(args, self.test_data, self.test_label, 'test')
        elif args.dataset == 'Lambada' or args.dataset == 'Lambada_test':
            self.train_dst = LambadaDataset_LLM(args, self.train_data, self.train_label, 'train')
            self.test_dst = LambadaDataset_LLM(args, self.test_data, self.test_label, 'test')
        elif args.dataset == 'MMLU':
            self.train_dst = MMLUDataset_LLM(args, self.train_data, self.train_label, 'train')
            self.test_dst = MMLUDataset_LLM(args, self.test_data, self.test_label, 'test')
        elif args.dataset == 'GMS8K' or args.dataset == 'GMS8K-test':
            self.train_dst = GSMDataset_LLM(args, self.train_data, self.train_label, 'train')
            self.test_dst = GSMDataset_LLM(args, self.test_data, self.test_label, 'test')
        elif args.dataset == 'MATH':
            self.train_dst = MATHDataset_LLM(args, self.train_data, self.train_label, 'train')
            self.test_dst = MATHDataset_LLM(args, self.test_data, self.test_label, 'test')
        elif args.dataset == 'CC_SBU':
            self.train_dst = CCSBUAlignDataset(args, self.train_data, self.train_label, self.vis_processors['train'],self.text_processors['train'], 'train')
            self.test_dst = CCSBUAlignDataset(args, self.test_data, self.test_label, self.vis_processors['eval'],self.text_processors['eval'],'test')
        elif args.dataset == 'TextVQA' or args.dataset == 'TextVQA-test':
            self.train_dst = TextVQADataset_train(args, self.train_data, self.train_label,self.vis_processors['train'],'train')
            self.test_dst = TextVQADataset_test(args, self.test_data, self.test_label,self.vis_processors['eval'],'test')
        else:
            self.train_dst = PassiveDataset_LLM(args, self.train_data, self.train_label, 'train')
            self.test_dst = PassiveDataset_LLM(args, self.test_data, self.test_label, 'test')
            
    def update_local_pred(self, pred):
        self.pred_received[self.args.k - 1] = pred

    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def cal_global_gradient_2slice(self, global_loss, global_pred):
        '''
        self.global_gradient = \partial global_loss / \partial global_pred
        '''
        # if (self.args.apply_mid == True and (self.index in self.args.defense_configs["party"])
        #       and (self.index < self.args.k - 1) and "tail" in self.mid_position):
        #     global_loss = global_loss + self.tail_mid_loss.to(global_loss.device)
       
        if self.args.task_type == 'QuestionAnswering':
            _gradients_start = torch.autograd.grad(global_loss, global_pred.start_logits, retain_graph=True)
            _gradients_end = torch.autograd.grad(global_loss, global_pred.end_logits, retain_graph=True)
            global_gradient = _gradients_end + _gradients_start
            global_gradient_clone = global_gradient[0].detach().clone()
            global_gradient_clone = global_gradient_clone / 2
            self.global_gradient = global_gradient_clone
        else:
            logits_gradients = torch.autograd.grad(global_loss, global_pred.logits, retain_graph=True)[0]
            if vfl_basic_config.num_of_slice==2:
                global_gradient_clone = logits_gradients.detach().clone()
            else:
                self.backward(2,retain_graph=True,gradient=logits_gradients)
                if self.input_tensors[2].grad is not None:
                    global_gradient_clone=self.input_tensors[2].grad.detach().clone()
                else:
                    global_gradient_clone=torch.autograd.grad(self.output_tensors[2],self.input_tensors[2],grad_outputs=logits_gradients,retain_graph=True)[0]
            self.global_gradient = global_gradient_clone

        self.global_gradient = self.apply_defense_on_global_gradient(self.global_gradient)

        return self.global_gradient

    def cal_global_gradient_3slice(self, global_loss, global_intermediate):
        '''
        self.global_gradient = \partial global_loss / \partial global_intermediate
        self.input_tensors[2] = global_intermediate model body output/model tail input
        '''
        # if (self.args.apply_mid == True and (self.index in self.args.defense_configs["party"])
        #       and (self.index < self.args.k - 1) and "tail" in self.mid_position):
        #     global_loss = global_loss + self.tail_mid_loss.to(self.global_loss.device)
        
        global_gradient_clone=torch.autograd.grad(global_loss,global_intermediate,retain_graph=True)[0]
        self.global_gradient = global_gradient_clone
        self.global_gradient = self.apply_defense_on_global_gradient(self.global_gradient)

        return self.global_gradient

    def apply_defense_on_global_gradient(self, origin_global_gradient):
        ######### Defense Applied on Global Gradients ###########
        if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"]))\
            and ('tail' in self.ad_position):
            self.origin_global_gradient = origin_global_gradient.clone()
            final_global_gradient = self.tail_adversarial_model(origin_global_gradient)
            return final_global_gradient
        ######### Defense Applied on Global Gradients ###########
        else:
            return origin_global_gradient

    def process_received_result(self,final_output):
        '''
        apply possible defenses on the final_output
        '''
        ##### Defense on received pred #### relevant decode/denoise method
        if self.args.apply_snd and self.args.vfl_model_slice_num == 2 and self.index in self.args.defense_configs["party"]:
            if self.args.model_architect == 'TQA':
                print('Warining: SnD currently do not supported for TQA')
            else:
                # print('2-slice Passive Party Denoise:',type(final_output),final_output.logits.shape)
                origin_device = final_output.logits.device
                final_output.logits = self.denoise_mod(self.output_tensors[0], self.snd_noise, final_output.logits, self.local_data_input['attention_mask']).to(origin_device)
        ##### Defense on received pred #### relevant decode/denoise method
        return final_output
    
    
    def cal_loss(self, pred, test=False):
        
        
        gt_one_hot_label = self.gt_one_hot_label  # label
        # ########### Normal Loss ###############
        if self.args.model_architect == 'CLS':  # self.args.task_type == 'SequenceClassification':
            pooled_logits = pred.logits # [bs, num_labels]
            ######### Defense ###########
            if self.args.vfl_model_slice_num == 2 and self.args.apply_mid \
            and (self.index in self.args.defense_configs["party"]) and ("tail" in self.mid_position):
                # print('== 2 slice mid')
                pooled_logits, self.tail_mid_loss = self.tail_mid_model(pooled_logits)  # , self.local_attention_mask
            ######### Defense ###########
        
            labels = torch.argmax(gt_one_hot_label,-1) # [bs, num_labels] -> [bs]
            
            # copied from transformer.model.gpt2.modeling_gpt
            if self.num_labels == 1:
                self.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
            else:
                self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        elif self.args.model_architect == 'CLM':  # self.args.task_type == 'CausalLM':
            lm_logits = pred.logits  # [bs, seq_len, vocab_size]
            ######### Defense ###########
            if self.args.vfl_model_slice_num == 2 and self.args.apply_mid \
            and (self.index in self.args.defense_configs["party"]) and ("tail" in self.mid_position):
                # print('== 2 slice mid')
                lm_logits, self.tail_mid_loss = self.tail_mid_model(lm_logits)  # , self.local_attention_mask
            ######### Defense ###########
            labels = torch.tensor(gt_one_hot_label).to(lm_logits.device)
            


            if len(labels.shape) > 1:
                # print('labels:',labels.shape,'  lm_logits:',lm_logits.shape)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.args.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Ensure tensors are on the same device
                shift_labels = shift_labels.to(shift_logits.device)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

            else:
                next_token_logits = lm_logits[:, -1, :]
                # print('next_token_logits:',next_token_logits.shape)
                # print('labels:',labels.shape)
                labels = torch.tensor(labels.long()).to(self.args.device)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss_fct(next_token_logits, labels)
                # print('loss:', loss)

        elif self.args.model_architect == 'MM':  # self.args.task_type == 'CausalLM':
            try:
                gt_one_hot_label = self.processed_labels
            except:
                pass
            
            logits = pred.logits  # [bs, seq_len, vocab_size]
            labels = torch.tensor(gt_one_hot_label).to(logits.device) # [bs, seq_len]
            
            if len(labels.shape) > 1:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.args.model_config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                # print('cal loss:',loss)

            else:
                next_token_logits = logits[:, -1, :]
                labels = torch.tensor(labels.long()).to(self.args.device)
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss_fct(next_token_logits, labels)
                # print('loss:', loss)

        elif self.args.model_architect == 'TQA':  # self.args.task_type == 'QuestionAnswering':
            start_logits = pred.start_logits
            end_logits = pred.end_logits

            # print('---- gt_one_hot_label -----') 
            # print(gt_one_hot_label) # [ [gold_s, gold_e]*bs ]

            golden_start_positions = torch.tensor([gt_one_hot_label[i][0] for i in range(len(gt_one_hot_label))])
            golden_end_positions = torch.tensor([gt_one_hot_label[i][1] for i in range(len(gt_one_hot_label))])

            golden_start_positions = golden_start_positions.long().to(start_logits.device)  # .unsqueeze(0)
            golden_end_positions = golden_end_positions.long().to(end_logits.device)

            loss = None

            if len(golden_start_positions.size()) > 1:
                golden_start_positions = golden_start_positions.squeeze(-1).to(start_logits.device)
            if len(golden_end_positions.size()) > 1:
                golden_end_positions = golden_end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            
            ignored_index = start_logits.size(1)  #print('ignored_index:',ignored_index)
            golden_start_positions = golden_start_positions.clamp(0, ignored_index)
            golden_end_positions = golden_end_positions.clamp(0, ignored_index)

            # print('start_logits end_logits:',start_logits.shape, end_logits.shape)
            # print('after clamp golden_start_positions golden_end_positions:')
            # print(golden_start_positions.shape, golden_start_positions)
            # print(golden_end_positions.shape, golden_end_positions)
            # print('-'*100)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, golden_start_positions)
            end_loss = loss_fct(end_logits, golden_end_positions)
            loss = (start_loss + end_loss) / 2

            # print('start_loss:',start_loss,' end_loss:',end_loss,' loss:',loss)

        else:
            assert 1 > 2, 'Task type not supported'

        self.global_loss = loss

        # ########### Defense on Loss ###############
        if self.args.apply_adversarial and (self.index in self.args.defense_configs["party"]) and \
            ('head' in self.ad_position):
            intermediate = self.output_tensors[0]  
            
            adversary_recovered_embedding = self.head_imagined_adversary(intermediate) # bs, seq_len, embed_dim
            real_embedding = self.local_model.embedding_output  # bs, seq_len, embed_dim

            if intermediate.shape != real_embedding.shape:
                real_embedding.transpose(0,1)
            if intermediate.shape != adversary_recovered_embedding.shape:
                adversary_recovered_embedding.transpose(0,1)
            
            self.head_adversary_attack_loss = self.adversary_crit(adversary_recovered_embedding.to(real_embedding.device), real_embedding) / \
                                         intermediate.shape[0]

            # avrage mapping distance on bs*seq_len   self.origin_pred: bs, seq_len, embed_dim
            self.head_mapping_distance = torch.norm(self.origin_pred - self.output_tensors[0], p=2) / (
                    self.origin_pred.shape[0] * self.origin_pred.shape[1])

            # print(f'main_loss={self.global_loss},mapping_distance={self.mapping_distance},adversary_attack_loss={self.adversary_attack_loss}')

            # renew global loss function : loss used to update adversarial model mapping
            self.head_adversarial_model_loss = self.adversary_lambda * self.head_mapping_distance.to(self.global_loss.device) - self.head_adversary_attack_loss.to(self.global_loss.device)
            self.global_loss = self.global_loss + self.head_adversarial_model_loss
        
        elif self.args.apply_mid == True and (self.index in self.args.defense_configs['party']):
            if ("tail" in self.mid_position):
                self.global_loss = self.global_loss + self.tail_mid_loss.to(self.global_loss.device)
        
        elif self.args.apply_textobfuscator == True and (self.index in self.args.defense_configs['party']):
            valid_ids = (self.local_data_input['attention_mask']==1) & (self.local_data_input['input_ids']!=2) & (self.local_data_input['input_ids']!=0) # [bs, seq_len]
            client_hidden_states = self.output_tensors[0][valid_ids]
            # self.origin_pred[valid_ids] # valid model head output [id_num, embed_dim]
            cluster_ids = torch.tensor([ [  self.token2cluster[ids.item()] if (ids.item() in self.token2cluster.keys()) else 0  for ids in batch_ids] for batch_ids in self.local_data_input['input_ids']], device=self.local_data_input['input_ids'].device)
            cluster_ids = cluster_ids[valid_ids] # [id_num]
            privacy_loss_dict = self.obfuscator_privacy_loss_func(client_hidden_states, cluster_ids)
            self.obfuscator_privacy_loss = 0
            if privacy_loss_dict != None:
                item_num = 0
                for loss_item in privacy_loss_dict.values():
                    item_num = item_num + 1
                    self.obfuscator_privacy_loss += loss_item
            self.obfuscator_privacy_loss = self.args.loss_lambda * self.obfuscator_privacy_loss
            self.global_loss = self.global_loss + self.obfuscator_privacy_loss.to(self.global_loss.device)
        # ########### Defense on Loss ###############

        return self.global_loss
    
    def update_loss_with_defense(self):
        # ########### Defense on Loss ###############
        if self.args.apply_adversarial and (self.index in self.args.defense_configs["party"]) and \
            ("tail" in self.ad_position):
            intermediate_gradient = self.global_gradient # gradient transmitted from model tail
            # print(f'intermediate_gradient:{intermediate_gradient.shape}')
            
            real_label = self.gt_one_hot_label  # bs, seq_len, embed_dim
            # print(f'real_label:{real_label.shape}')
            adversary_recovered_label = self.tail_imagined_adversary(intermediate_gradient) # bs, seq_len, embed_dim

            self.tail_adversary_attack_loss = self.adversary_crit(adversary_recovered_label, real_label)
            # avrage mapping distance on bs*seq_len   self.origin_global_gradient: bs, seq_len, embed_dim
            self.tail_mapping_distance = torch.norm(self.origin_global_gradient - self.global_gradient, p=2) / (
                    self.global_gradient.shape[0] * self.global_gradient.shape[1])

            # renew global loss function : loss used to update adversarial model mapping
            self.tail_adversarial_model_loss = self.adversary_lambda * self.tail_mapping_distance.to(self.global_loss.device) - self.tail_adversary_attack_loss.to(self.global_loss.device)
            self.global_loss = self.global_loss + self.tail_adversarial_model_loss
        # ########### Defense on Loss ###############

        return 


    def gradient_calculation(self, pred_list, loss):
        pred_gradients_list = []
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients_list.append(torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True))
            pred_gradients_list_clone.append(pred_gradients_list[ik][0].detach().clone())
        # self.global_backward(pred, loss)
        return pred_gradients_list, pred_gradients_list_clone

    def update_local_gradient(self, gradient):
        self.local_gradient = gradient

    def global_LR_decay(self, i_epoch, is_return=False):
        eta_0 = self.args.main_lr
        eta_t = eta_0 / (np.sqrt(i_epoch + 1))
        if is_return:
            return eta_t
        if self.global_model_optimizer != None:
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t

    @timer()
    def give_pred(self, use_cache=False):
        self.local_data_input['use_cache'] = use_cache
        if use_cache:
            self.local_data_input['past_key_values'] = self.past_key_values.get(0)

        ######### Defense Applied on Text Input ###########
        if self.is_first_forward_iter:
            if self.args.apply_custext and (self.index in self.args.defense_configs["party"]):
                input_device = self.local_data_input['input_ids'].device
                origin_len = self.local_data_input['input_ids'].shape[-1]
                # print("origin_input_ids:", self.local_data_input['input_ids'].shape)
                
                origin_input_sentence = []
                for row in self.local_data_input['input_ids']:
                    sentence = self.args.tokenizer.decode(row, skip_special_tokens=True)
                    origin_input_sentence.append(sentence)
                print("origin_input_sentence:", origin_input_sentence)
                
                sanitized_sentence = generate_new_sents_s1(origin_input_sentence,self.sim_word_dict,self.p_dict,save_stop_words=False, args = self.args)
                print("sanitized_sentence:", sanitized_sentence)
                
                # Convert sentence back to tensor
                tokenized_sanitized_sentence = self.args.tokenizer(list(sanitized_sentence), 
                                                padding=self.args.padding, truncation=self.args.truncation, \
                                                max_length=origin_len, return_tensors='pt',
                                                add_special_tokens=self.args.add_special_tokens)
                self.local_data_input['input_ids'] = tokenized_sanitized_sentence['input_ids']
                self.local_data_input['attention_mask'] = tokenized_sanitized_sentence['attention_mask']
                # print("sanitized_input_ids:", self.local_data_input['input_ids'].shape)
                # assert 1>2

            if self.args.apply_inferdpt and (self.index in self.args.defense_configs["party"]):
                input_device = self.local_data_input['input_ids'].device
                new_input_ids = []
                for original_input_ids in self.local_data_input['input_ids']: #[bs, seq_len]
                    assert self.args.epsilon > 0, "epsilon should be greater than 0"
                    tokens = self.args.tokenizer.convert_ids_to_tokens(original_input_ids)
                    # print('before tokens:',len(tokens))
                    # print(tokens)
                    new_tokens = []
                    Delta_u = 1.0  
                    exp_factor = self.args.epsilon / (2 * Delta_u)
                    for origin_token in tokens:
                        if origin_token[0] == ' ':
                            origin_token = origin_token[1:]
                        origin_embed = self.token_to_vector_dict.get(origin_token, None)
                        if origin_embed is None:
                            new_tokens.append(origin_token)
                            continue
                        noise_embed = add_laplace_noise_to_vector(origin_embed, self.args.epsilon, self.delta_f)
                        similarity = cosine_similarity_vectors(origin_embed, noise_embed)
                        sorted_distances_for_token = self.sorted_similarities.get(origin_token, None)
                        if sorted_distances_for_token is None:
                            continue
                        token_only = sorted_distances_for_token[0]
                        similarity_only = sorted_distances_for_token[1]
                        arr = np.flip(similarity_only)
                        index = np.searchsorted(arr, similarity)
                        index = len(arr) - index
                        close_tokens = token_only[:index]
                        close_similarities = similarity_only[:index]
                        if len(close_tokens) == 0:
                            continue
                        unnormalized_probabilities = np.exp(exp_factor * np.array(close_similarities))
                        total_unnormalized_prob = np.sum(unnormalized_probabilities)
                        probabilities = unnormalized_probabilities / total_unnormalized_prob
                        selected_token = np.random.choice(close_tokens, p=probabilities)
                        new_tokens.append(selected_token)
                    new_token_ids = self.args.tokenizer.convert_tokens_to_ids(new_tokens)
                    # sentence = self.args.tokenizer.decode(new_token_ids)
                    # print('after:',sentence)
                    new_input_ids.append(torch.tensor(new_token_ids))
                new_input_ids = torch.stack(new_input_ids) # [bs, seq_len]
                self.local_data_input['input_ids'] = new_input_ids.to(input_device)
        ######### Defense Applied on Text Input ###########
        
        # collect processed labels (only in some cases)
        # model 0 head  / model 1 body(active) / model 2 tail
        intermediate = self.forward(model_index=0, **self.local_data_input)
        

        if 'processed_labels' in intermediate.keys():
            self.processed_labels = intermediate['processed_labels']
            del(intermediate['processed_labels'])

        self.local_attention_mask = intermediate['attention_mask'] if ('attention_mask' in intermediate) else None
        self.local_pred_clone = self.output_tensors[0].detach().clone()
        if self.local_attention_mask != None:
            self.local_attention_mask = self.local_attention_mask.detach().clone()

        ######### Defense Applied on Local Model Prediction Process ###########
        if self.is_first_forward_iter:
            if self.args.apply_mid and (self.index in self.args.defense_configs["party"]) and\
             ("head" in self.mid_position):
                self.output_tensors[0], self.head_mid_loss = self.head_mid_model(self.output_tensors[0].to(next(self.mid_model.parameters()).device))  # , self.local_attention_mask
                self.local_pred_clone = self.output_tensors[0].detach().clone()
            
            elif (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])) \
                and ('head' in self.ad_position):
                self.origin_pred = self.output_tensors[0].clone().to(next(self.head_adversarial_model.parameters()).device)
                self.output_tensors[0] = self.head_adversarial_model(self.origin_pred)
                self.local_pred_clone = self.output_tensors[0].detach().clone()
            
            elif self.args.apply_textobfuscator == True and (self.index in self.args.defense_configs["party"]):
                self.origin_pred = self.output_tensors[0].clone()
                target_noise =  self.obfuscator.sample(self.output_tensors[0].shape).type_as(self.output_tensors[0])
                # self.output_tensors[0] =  self.output_tensors[0] + target_noise
                self.local_pred_clone = self.output_tensors[0].detach().clone() + target_noise

            elif self.args.apply_snd and (self.index in self.args.defense_configs["party"]):
                # print('Passive Party Add noise')
                self.snd_noise = sample_noise_Chi(self.output_tensors[0].shape, self.args.train_eta).to(self.output_tensors[0].device)
                # print('noise:',self.snd_noise.shape,'  origin:',self.output_tensors[0].shape)
                self.local_pred_clone = self.output_tensors[0].detach().clone() + self.snd_noise
        ######### Defense Applied on Local Model Prediction Process ###########

        self.output_attention_mask[0] = self.local_attention_mask

        intermediate['inputs_embeds'] = self.local_pred_clone
        if self.local_attention_mask != None:
            intermediate['attention_mask'] = self.local_attention_mask

        return intermediate
    
    def inferdpt_decode(self, original_prompt, pertrubed_answer):
        if self.args.dataset in ['GMS8K','GMS8K-test' ]:
            for _i in range(len(original_prompt)):
                instruction_match = re.search(r'### Instruction:(.*?)### Response:', original_prompt[_i], re.DOTALL)
                original_prompt[_i] = instruction_match.group(1).strip()
        decode_input = [self.decode_template.format(prefix=original_prompt[_i] ,perturbed_answer=pertrubed_answer[_i])\
            for _i in range(len(original_prompt))]
        
        print('===============')
        print('Extraction Input:')
        print(decode_input)
        
        
        decode_input = self.decode_model_tokenizer(decode_input,return_tensors='pt')
        self._tensor_to_device(decode_input, self.decode_model.device)
        extracted_answer = self.decode_model.generate(**decode_input,**self.args.decode_generation_kwargs)
        extracted_answer = extracted_answer[:,decode_input['input_ids'].shape[1]:]
        extracted_answer_txt = self.decode_model_tokenizer.decode(extracted_answer.squeeze().tolist())
        
        # convert to token ids correspond to self.args.tokenizer
        extracted_answer = self.args.tokenizer(extracted_answer_txt)['input_ids']
        extracted_answer_txt = self.args.tokenizer.decode(extracted_answer)
        print('----------------')
        print('Extraction Output:')
        print(extracted_answer_txt)
        
        extracted_answer = torch.tensor(extracted_answer).unsqueeze(0)
        
        # print(type(extracted_answer),extracted_answer.shape)
        # print(extracted_answer_txt)
        
        return extracted_answer
    
    def give_final_pred(self, resp):
        ######### Defense Applied on Local Model Tail Prediction Process ###########
        if self.args.vfl_model_slice_num == 3 and self.args.apply_mid and \
            (self.index in self.args.defense_configs["party"]) and ("tail" in self.mid_position):
            received_pred = resp['inputs_embeds'] 
            received_pred, self.tail_mid_loss = self.tail_mid_model(received_pred)  # , self.local_attention_mask
            resp['inputs_embeds'] = received_pred
            
        if self.args.vfl_model_slice_num == 3 and self.args.apply_snd and self.index in self.args.defense_configs["party"]:
            # print('3-slice Passive Party Denoise:')
            # resp['denoise_mod'] = self.denoise_mod
            # resp['snd_noise'] = self.snd_noise
            # resp['original_embedding'] = self.output_tensors[0]
            origin_device = resp['inputs_embeds'].device
            received_pred = resp['inputs_embeds']
            resp['inputs_embeds'] = self.denoise_mod(self.output_tensors[0], self.snd_noise, \
                received_pred, self.local_data_input['attention_mask']).to(origin_device)
        ######### Defense Applied on Local Model Tail Prediction Process ###########
        return self.forward(2, **resp)

    def local_backward(self):
        self.num_local_updates += 1  # another update

        ###### Update Model Head #########
        # adversarial training : update adversarial model
        if (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"]))\
            and ('head' in self.ad_position):
            # update imagined_adversary_model
            self.head_imagined_adversary_optimizer.zero_grad()
            self.head_adversary_attack_loss.backward(retain_graph=True)
            self.head_imagined_adversary_optimizer.step()

            self.local_model_optimizer.zero_grad()
            # local model trainable part
            local_model_params = list(self.head_adversarial_model.parameters())
            for param in self.local_model.parameters():
                if param.requires_grad:
                    local_model_params.append(param)

            self.local_gradient=self.local_gradient.to(self.output_tensors[0].device)

            weights_grad_a = torch.autograd.grad(
                self.output_tensors[0],
                local_model_params,
                grad_outputs=self.local_gradient,
                retain_graph=True,
                allow_unused=True
            )
            for w, g in zip(local_model_params, weights_grad_a):
                if w.requires_grad and g != None:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()

            weights_grad_a = torch.autograd.grad(
                self.head_adversarial_model_loss,
                self.head_adversarial_model.parameters(),
                retain_graph=True,
                allow_unused=True,
            )
            for w, g in zip(self.head_adversarial_model.parameters(), weights_grad_a):
                if w.requires_grad:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()

            self.local_model_optimizer.step()
        
        elif (self.args.apply_mid == True and (self.index in self.args.defense_configs["party"])
              and (self.index < self.args.k - 1) and "head" in self.mid_position):
            self.local_model_optimizer.zero_grad()  # self.mid_model_optimizer.zero_grad()

            # update mid_model+local_model with mid_loss
            self.head_mid_loss.backward(retain_graph=True)
            
            # update mid_model and local model with global_loss
            self.local_gradient = self.local_gradient.to(self.output_tensors[0].device)
            weights_grad_a = torch.autograd.grad(
                self.output_tensors[0],
                self.head_mid_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True,
                allow_unused=True,
            )
            for w, g in zip(self.head_mid_model.parameters(), weights_grad_a):
                if w.requires_grad:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()

            local_model_params = []
            for param in self.local_model.parameters():
                if param.requires_grad:
                    local_model_params.append(param)
            if len(local_model_params) > 0:
                self.weights_grad_a = torch.autograd.grad(
                    self.output_tensors[0],
                    local_model_params,
                    grad_outputs=self.local_gradient,
                    retain_graph=True,
                    allow_unused=True,
                )
                for w, g in zip(local_model_params, self.weights_grad_a):
                    if w.requires_grad and g != None:
                        if w.grad != None:
                            w.grad += g.detach()
                        else:
                            w.grad = g.detach()
            
            self.local_model_optimizer.step()
            
        elif (self.args.apply_textobfuscator == True and (self.index in self.args.defense_configs["party"])
              and (self.index < self.args.k - 1)):
            
            self.local_model_optimizer.zero_grad()  # self.mid_model_optimizer.zero_grad()

            local_model_params = []
            for param in self.local_model.parameters():
                if param.requires_grad:
                    local_model_params.append(param)
            
            if len(local_model_params) > 0:
                # update local model with obfuscator_privacy_loss
                self.obfuscator_privacy_loss.backward(retain_graph=True)
                
                # update local model with global loss
                self.local_gradient = self.local_gradient.to(self.output_tensors[0].device)
                self.weights_grad_a = torch.autograd.grad(
                    self.output_tensors[0],
                    local_model_params,  # self.local_model.parameters()
                    grad_outputs=self.local_gradient,
                    retain_graph=True,
                    allow_unused=True,
                )
                for w, g in zip(local_model_params, self.weights_grad_a):
                    if w.requires_grad and g != None:
                        if w.grad != None:
                            w.grad += g.detach()
                        else:
                            w.grad = g.detach()

            self.local_model_optimizer.step()

            # update 
        else:  # W/O Defense
            if self.local_model_optimizer != None:
                self.local_model_optimizer.zero_grad()


                # local model trainable part
                # local_model_params = list(filter(lambda x: x.requires_grad, self.local_model.parameters()))
                local_model_params = []
                for param in self.local_model.parameters():
                    if param.requires_grad:
                        local_model_params.append(param)

                self.local_gradient = self.local_gradient.to(self.output_tensors[0].device)

                if len(local_model_params) > 0:
                    self.weights_grad_a = torch.autograd.grad(
                        self.output_tensors[0],
                        local_model_params,  # self.local_model.parameters()
                        grad_outputs=self.local_gradient,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    for w, g in zip(local_model_params, self.weights_grad_a):
                        if w.requires_grad and g != None:
                            if w.grad != None:
                                w.grad += g.detach()
                            else:
                                w.grad = g.detach()

                    self.local_model_optimizer.step()
        
        ###### Update Model Tail #########
        if self.args.vfl_model_slice_num== 3 and self.local_model_tail_optimizer != None:
            self.local_model_tail_optimizer.zero_grad()

            # local model tail trainable part
            local_model_tail_params = []
            for param in self.local_model_tail.parameters():
                if param.requires_grad:
                    local_model_tail_params.append(param)

            ### Defense        
            if (self.args.apply_mid == True and (self.index in self.args.defense_configs["party"])
              and (self.index < self.args.k - 1) and "tail" in self.mid_position):
                
                mid_weights_grad_a = torch.autograd.grad(
                    self.global_loss, #final_loss,
                    self.tail_mid_model.parameters(),
                    retain_graph=True,
                    allow_unused=True,
                )
                for w, g in zip(self.tail_mid_model.parameters(), mid_weights_grad_a):
                    if w.requires_grad:
                        if w.grad != None:
                            w.grad += g.detach()
                        else:
                            w.grad = g.detach()

                if len(local_model_tail_params) > 0:
                    self.weights_grad_a_tail = torch.autograd.grad(
                        self.global_loss, # model tail output, the final pred
                        local_model_tail_params,  # self.local_model.parameters()
                        retain_graph=True,
                        allow_unused=True,
                    )
                    for w, g in zip(local_model_tail_params, self.weights_grad_a_tail):
                        if w.requires_grad and g != None:
                                if w.grad != None:
                                    w.grad += g.detach()
                                else:
                                    w.grad = g.detach()
                    
                self.local_model_tail_optimizer.step()
            
            elif (self.args.apply_adversarial == True and (self.index in self.args.defense_configs["party"])) \
                    and ("tail" in self.ad_position):
                # update imagined_adversary_model
                self.tail_imagined_adversary_optimizer.zero_grad()
                self.tail_adversary_attack_loss.backward(retain_graph=True)
                self.tail_imagined_adversary_optimizer.step()


                self.local_model_tail_optimizer.zero_grad()

                # update local model tail trainable part
                local_model_tail_params.extend(list(self.tail_adversarial_model.parameters()))

                weights_grad_a = torch.autograd.grad(
                    self.global_loss,
                    local_model_tail_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                for w, g in zip(local_model_tail_params, weights_grad_a):
                    if w.requires_grad and g != None:
                        if w.grad != None:
                            w.grad += g.detach()
                        else:
                            w.grad = g.detach()

                self.local_model_tail_optimizer.step()

            ### W/O Defense
            else:
                if len(local_model_tail_params) > 0:
                    self.weights_grad_a_tail = torch.autograd.grad(
                        self.global_loss, # model tail output, the final pred
                        local_model_tail_params,  # self.local_model.parameters()
                        retain_graph=True,
                        allow_unused=True,
                    )
                    for w, g in zip(local_model_tail_params, self.weights_grad_a_tail):
                        if w.requires_grad and g != None:
                            if w.grad != None:
                                w.grad += g.detach()
                            else:
                                w.grad = g.detach()

                    self.local_model_tail_optimizer.step()

    def launch_defense(self, gradients_list, _type):
            if _type == 'gradients':
                return apply_defense(self.args, _type, gradients_list)
            elif _type == 'pred':
                return apply_defense(self.args, _type, gradients_list)
            else:
                # further extention
                return gradients_list
