import sys, os
sys.path.append(os.pardir)
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import numpy as np
import copy
import pickle 
import matplotlib.pyplot as plt
import itertools 
from scipy import optimize
import cv2

from evaluates.attacks.attacker import Attacker
from models.global_models import *
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from dataset.party_dataset import *

from evaluates.defenses.defense_functions import LaplaceDP_for_pred,GaussianDP_for_pred
from models.llm_models.processors.blip_processors import *

def label_to_one_hot(target, num_classes=10):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        # print("use target itself", target.size())
        onehot_target = target.type(torch.float32)
    except:
        target = torch.unsqueeze(target, 1)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes)
        onehot_target.scatter_(1, target, 1)
    return onehot_target

class custom_AE(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super(custom_AE,self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 600), 
            nn.LayerNorm(600),
            nn.ReLU(),
            
            nn.Linear(600, 200), 
            nn.LayerNorm(200),
            nn.ReLU(),
            
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            
            nn.Linear(100, target_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.tensor(x,dtype=torch.float32)
        return self.net(x)


class VanillaModelInversion_WhiteBox_test(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        # 
        self.attack_name = "VanillaModelInversion_WhiteBox_test"
        self.args = args
        self.top_vfl = top_vfl
        self.vfl_info = top_vfl.final_state

        # self.vfl_first_epoch = top_vfl.first_epoch_state
        
        # prepare parameters
        self.task_type = args.task_type

        self.device = args.device
        self.num_classes = args.num_classes
        self.label_size = args.num_classes
        self.k = args.k
        self.batch_size = args.batch_size

        # attack configs
        self.party = args.attack_configs['party'] # parties that launch attacks , default 1(active party attack)
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.attack_batch_size = args.attack_configs['batch_size']
        self.attack_sample_num = args.attack_configs['attack_sample_num']
        
        if 'loss_type' not in args.attack_configs.keys():
            self.loss_type = 'cross_entropy'
        else:
            self.loss_type = args.attack_configs['loss_type'] # mse/cross_entropy
        
        if self.loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
   
    
    def set_seed(self,seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def attack(self):
        self.set_seed(123)
        print_every = 1

        for attacker_ik in self.party: # attacker party #attacker_ik
            assert attacker_ik == (self.k - 1), 'Only Active party launch input inference attack'

            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)
            index = attacker_ik

            # collect necessary information
            local_model = self.top_vfl.parties[0].local_model.to(self.device)
            #self.vfl_info['local_model_head'].to(self.device) # Passive
            local_model.eval()

            if self.args.model_architect == 'MM':
                vis_processor = self.top_vfl.parties[0].vis_processors['eval']
                #self.vfl_info['vis_processors']['eval']
                reverse_vis_processor = MiniCPM_Transform_reverse()

            batch_size = self.attack_batch_size

            attack_result = pd.DataFrame(columns = ['Pad_Length','Length','Precision', 'Recall','Img_Mse','Rand_Img_Mse'])

            # attack_test_dataset = self.top_vfl.parties[0].test_dst
            test_data = self.vfl_info["test_data"][0] 
            test_label = self.vfl_info["test_label"][0] 
            
            if len(test_data) > self.attack_sample_num:
                test_data = test_data[:self.attack_sample_num]
                test_label = test_label[:self.attack_sample_num]
                # attack_test_dataset = attack_test_dataset[:self.attack_sample_num]
            
            if self.args.dataset == 'Lambada' or self.args.dataset == 'Lambada_test':
                attack_test_dataset = LambadaDataset_LLM(self.args, test_data, test_label, 'test')
            elif self.args.dataset == 'TextVQA' or self.args.dataset == 'TextVQA-test':
                attack_test_dataset = TextVQADataset_train(self.args, test_data, test_label, vis_processor,'train')
            elif self.args.dataset == 'GMS8K' or self.args.dataset == 'GMS8K-test':
                attack_test_dataset = GSMDataset_LLM(self.args, test_data, test_label, 'test')
            else:
                attack_test_dataset = PassiveDataset_LLM(self.args, test_data, test_label)

            attack_info = f'Attack Sample Num:{len(attack_test_dataset)}'
            print(attack_info)
            append_exp_res(self.args.exp_res_path, attack_info)

            test_data_loader = DataLoader(attack_test_dataset, batch_size=batch_size ,collate_fn=lambda x:x ) # ,
            del(self.vfl_info)
            # test_data_loader = self.vfl_info["test_loader"][0] # Only Passive party has origin input
            
            flag = 0
            enter_time = time.time()
            img_id = 0
            for origin_input in test_data_loader:
                ## origin_input: list of bs * (input_discs, label)
                batch_input_dicts = []
                batch_label = []
                for bs_id in range(len(origin_input)):
                    # Input Dict
                    batch_input_dicts.append(origin_input[bs_id][0])
                    # Label
                    if type(origin_input[bs_id][1]) != str:
                        batch_label.append(origin_input[bs_id][1].tolist())
                    else:
                        batch_label.append(origin_input[bs_id][1])

                data_inputs = {}
                for key_name in batch_input_dicts[0].keys():
                    if isinstance(batch_input_dicts[0][key_name], torch.Tensor):
                        data_inputs[key_name] = torch.stack( [batch_input_dicts[i][key_name] for i in range(len(batch_input_dicts))] )
                    else:
                        data_inputs[key_name] = [batch_input_dicts[i][key_name] for i in range(len(batch_input_dicts))]         

                print('VMI data_inputs:',data_inputs.keys())
                # print('image_bound:',data_inputs['image_bound']) # list of 4 * tensor(1, 2)
                # print('data_inputs input_ids:',data_inputs['input_ids'].shape) #bs,160
                # print(self.args.tokenizer.decode(data_inputs['input_ids'][0]))
                # print('data_inputs pixel_values:')# list of bs * list of 1 * torch.size(3, 903, 1024)
                # print(data_inputs['pixel_values'][0][0].size()) #1,160

                # real received intermediate result
                self.top_vfl.parties[0].obtain_local_data(data_inputs)
                self.top_vfl.parties[0].gt_one_hot_label = batch_label

                all_pred_list = self.top_vfl.pred_transmit()
                real_results = all_pred_list[0]
                self.top_vfl._clear_past_key_values()


                # each sample in a batch
                for _id in range(len(origin_input)):
                    img_id = img_id + 1

                    ## origin data
                    sample_origin_data = batch_input_dicts[_id]['input_ids'].unsqueeze(0) # [1,sequence length]
                    if self.args.model_architect == 'MM':
                        sample_origin_image_list = batch_input_dicts[_id]['pixel_values']# list of 1 * [3, img_size, 1024]
                        sample_origin_image_bound = batch_input_dicts[_id]['image_bound'].unsqueeze(0)
                        sample_image_id = batch_input_dicts[_id]['pixel_values']
                    bs, seq_length = sample_origin_data.shape
                    
                    ## received info
                    if real_results['inputs_embeds'].shape[1] != seq_length:
                        received_intermediate = real_results['inputs_embeds'].transpose(0,1)[_id].unsqueeze(0) # [1,256,768]
                    else:
                        received_intermediate = real_results['inputs_embeds'][_id].unsqueeze(0) # [1,256,768]

                    ## initial guess
                    if hasattr(real_results,'attention_mask'):
                        received_attention_mask = real_results['attention_mask'][_id].unsqueeze(0) # [1,256]
                    else:
                        received_attention_mask = None
                    if received_attention_mask != None:
                        dummy_attention_mask = received_attention_mask.to(self.device)
                    else:
                        dummy_attention_mask = None
                    if 'token_type_ids' in batch_input_dicts[0].keys():
                        dummy_local_batch_token_type_ids = batch_input_dicts[_id]['token_type_ids'].unsqueeze(0).to(self.device)
                    else:
                        dummy_local_batch_token_type_ids = None
                    
                    if not self.args.model_architect == 'MM':
                        dummy_embedding = torch.zeros([bs,seq_length,self.args.model_embedded_dim]).type(torch.float32).to(self.device)
                        dummy_embedding.requires_grad_(True) 
                        optimizer = torch.optim.Adam([dummy_embedding], lr=self.lr)
                        
                        def get_cost(dummy_embedding):
                            # compute dummy result
                            dummy_input = {
                            'input_ids':None, 'attention_mask':dummy_attention_mask,\
                            'inputs_embeds':dummy_embedding, 'token_type_ids':dummy_local_batch_token_type_ids
                            }
                            dummy_intermediate_dict = local_model(**dummy_input)
                            local_model._clear_past_key_values()

                            dummy_intermediate = dummy_intermediate_dict.get('inputs_embeds')
                            if dummy_intermediate.shape[1] != seq_length:
                                dummy_intermediate = dummy_intermediate.transpose(0,1)
                            # print('dummy_intermediate:',dummy_intermediate.shape) # 1, seq_len, embed_dim
                            # print('received_intermediate:',received_intermediate.shape)
                        
                            crit = self.criterion #nn.MSELoss() #nn.CrossEntropyLoss()
                            _cost = crit(dummy_intermediate, received_intermediate)
                            return _cost
            
                    else:
                        dummy_vllm_embedding = ( 
                            local_model.llm.model.embed_tokens(sample_origin_data) * local_model.llm.config.scale_emb
                        )
                        dummy_vllm_embedding.requires_grad = False
                        # torch.zeros([bs,seq_length,self.args.model_embedded_dim]).type(torch.float32).to(self.device)

                        dummy_image_list = [torch.rand(sample_origin_image.size()).type(torch.float32).to(self.device)
                            for sample_origin_image in sample_origin_image_list]
                        for dummy_img in dummy_image_list:
                            dummy_img.requires_grad = True

                        optimizer = torch.optim.Adam(dummy_image_list, lr=self.lr) # +[dummy_vllm_embedding]
                        # optimizer = torch.optim.Adam(dummy_image_list, lr=self.lr,
                        #     betas=(0.9, 0.999), eps=1e-08,weight_decay=0, amsgrad=False)
                        def get_cost(dummy_image_list):
                            # compute dummy result
                            embedding_dict = {"pixel_values":[dummy_image_list], # bs, img_size, embed_dim2304 [image]
                                "vllm_embedding":dummy_vllm_embedding, # bs, seq_len, embed_dim2304 [text]
                                "image_bound":sample_origin_image_bound}
                            dummy_embedding = local_model.get_vllm_embedding(
                                data = embedding_dict 
                            )[0] # 1, 160, 2304

                            dummy_input = {
                            'input_ids':None, 'attention_mask':dummy_attention_mask,\
                            'inputs_embeds':dummy_embedding, 'token_type_ids':dummy_local_batch_token_type_ids
                            }
                            dummy_intermediate_dict = local_model(**dummy_input)
                            local_model._clear_past_key_values()

                            dummy_intermediate = dummy_intermediate_dict.get('inputs_embeds')
                            if dummy_intermediate.shape[1] != seq_length:
                                dummy_intermediate = dummy_intermediate.transpose(0,1)
                            # print('dummy_intermediate:',dummy_intermediate.shape) # 1, seq_len, embed_dim
                            # print('received_intermediate:',received_intermediate.shape)
                        
                            crit = self.criterion #nn.CrossEntropyLoss()
                            _cost = crit(dummy_intermediate, received_intermediate)
                            return _cost
            
                        
                    
                    cost_function = torch.tensor(10000000)
                    last_cost = torch.tensor(10000000)
                    _iter = 0
                    while _iter<self.epochs: # cost_function.item()>=0.1 and 
                        optimizer.zero_grad()
                        if not self.args.model_architect == 'MM':
                            cost_function = get_cost(dummy_embedding)
                        else:
                            cost_function = get_cost(dummy_image_list)
                        cost_function.backward()
                        optimizer.step()

                        _iter+=1
                        if _iter%20 == 0:
                            print('=== iter ',_iter,'  cost:',cost_function.item())
                        
                        #     if last_cost < cost_function.item():
                        #         break
                            
                        # last_cost = min(cost_function.item(),last_cost)
                    
                    # recover tokens from dummy embeddings
                    if not self.args.model_architect == 'MM':
                        dummy_embedding = dummy_embedding.squeeze()
                        # print('dummy_embedding:',dummy_embedding.shape)

                        predicted_indexs = []
                        for i in range(dummy_embedding.shape[0]):
                            _dum = dummy_embedding[i]
                            # print(_dum.unsqueeze(0).shape)
                            if self.args.model_type  in ['Bert','Roberta']:
                                cos_similarities = nn.functional.cosine_similarity\
                                                (local_model.embeddings.word_embeddings.weight, _dum.unsqueeze(0), dim=1) # .unsqueeze(0)
                            else:
                                cos_similarities = nn.functional.cosine_similarity\
                                                    (local_model.get_input_embeddings().weight, _dum.unsqueeze(0), dim=1) # .unsqueeze(0)
                                
                            # print('cos_similarities:',cos_similarities.shape)
                            _, predicted_index = cos_similarities.max(0)
                            predicted_index = predicted_index.item()
                            predicted_indexs.append(predicted_index)
                    else:
                        dummy_vllm_embedding = dummy_vllm_embedding.squeeze()
                        predicted_indexs = []
                        for i in range(dummy_vllm_embedding.shape[0]):
                            _dum = dummy_vllm_embedding[i]
                            # print(_dum.unsqueeze(0).shape) # 1, 2304
                            # print(local_model.get_input_embeddings().weight.shape) # 122753, 2304  
                            if self.args.model_type  in ['Bert','Roberta']:
                                cos_similarities = nn.functional.cosine_similarity\
                                                (local_model.embeddings.word_embeddings.weight, _dum.unsqueeze(0), dim=1) # .unsqueeze(0)
                            else:
                                cos_similarities = nn.functional.cosine_similarity\
                                                    (local_model.get_input_embeddings().weight, _dum.unsqueeze(0), dim=1) # .unsqueeze(0)
                                
                            # print('cos_similarities:',cos_similarities.shape)
                            _, predicted_index = cos_similarities.max(0)
                            predicted_index = predicted_index.item()
                            predicted_indexs.append(predicted_index)
                        
                    sample_origin_id = sample_origin_data.squeeze().tolist()
                    origin_text = self.args.tokenizer.decode(sample_origin_id)
                    clean_sample_origin_id = sample_origin_id.copy()
                    while self.args.tokenizer.pad_token_id in clean_sample_origin_id:
                        clean_sample_origin_id.remove(self.args.tokenizer.pad_token_id) # with no pad
                    
                    suc_cnt = 0
                    for _sample_id in clean_sample_origin_id:
                        if _sample_id in predicted_indexs:
                            suc_cnt+=1
                    recall = suc_cnt / len(clean_sample_origin_id)
                    suc_cnt = 0
                    common_ids = []
                    for _pred_id in predicted_indexs:
                        if _pred_id in clean_sample_origin_id:
                            common_ids.append(_pred_id)
                            suc_cnt+=1
                    precision = suc_cnt / len(predicted_indexs)
                    
                    pred_text = self.args.tokenizer.decode(predicted_indexs)
                    if flag == 0:
                        print('len:',len(clean_sample_origin_id),'  precision:',precision, ' recall:',recall)
                        print('origin_text:\n',origin_text)
                        print('-'*25)
                        print('pred_text:\n',pred_text)
                        print('-'*25)
                    flag += 1

                    # recover images from dummy embeddings
                    mse = None
                    rand_mse = None
                    if self.args.model_architect == 'MM':
                        img_criterion = nn.MSELoss()
                        
                        img_mse = img_criterion(sample_origin_image_list[0],dummy_image_list[0]).item()
                        
                        rand_image = torch.randn(sample_origin_image_list[0].size()).to(self.device)
                        rand_mse = img_criterion(sample_origin_image_list[0], rand_image).item()
                        
                        def save_tensor_to_image(input_tensor,img_dir ,image_name):
                            img = reverse_vis_processor(input_tensor)
                            img.save(img_dir+f'{image_name}.jpg')
                            print('save:',img_dir+f'{image_name}.jpg')
                        img_dir = f"{self.args.exp_res_dir}/{self.attack_name}/"
                        if not os.path.exists(img_dir):
                            os.makedirs(img_dir)
                        save_tensor_to_image(dummy_image_list[0],img_dir, f'dummy_{img_id}')
                        save_tensor_to_image(sample_origin_image_list[0],img_dir,f'origin_{img_id}')

                        # print('img_mse:',img_mse,'  rand_mse:',rand_mse)

                    attack_result.loc[len(attack_result)] = [len(sample_origin_id), len(clean_sample_origin_id), precision,recall,img_mse,rand_mse ]

                    
                    if not self.args.model_architect == 'MM':
                        del(dummy_embedding)
                    else:
                        del(dummy_vllm_embedding)
                        del(dummy_image_list)
                    del(dummy_attention_mask)

            end_time = time.time()
        
        attack_total_time = end_time - enter_time
        Precision = attack_result['Precision'].mean()
        Recall = attack_result['Recall'].mean()
        Image_Mse = attack_result['Img_Mse'].mean()
        Rand_Image_Mse = attack_result['Rand_Img_Mse'].mean()


        print('FINAL: Image_Mse:',Image_Mse,'  Rand_Image_Mse:',Rand_Image_Mse,'  Recall:',Recall)

        
        

        # model_name = self.args.model_list['0']['type']
        # if self.args.pretrained == 1:
        #     result_path = f'./exp_result/{str(self.args.dataset)}/{self.attack_name}/{self.args.defense_name}_{self.args.defense_param}_pretrained_{str(model_name)}/'
        # else:
        #     result_path = f'./exp_result/{str(self.args.dataset)}/{self.attack_name}/{self.args.defense_name}_{self.args.defense_param}_finetuned_{str(model_name)}/'

        # if not os.path.exists(result_path):
        #     os.makedirs(result_path)
        # result_file_name = result_path + f'{self.args.pad_info}_{str(Precision)}_{str(Recall)}.csv'
        # print(result_file_name)
        # attack_result.to_csv(result_file_name)

        
        return Precision, Recall, Image_Mse, Rand_Image_Mse, attack_total_time
        # else:
        #     return {
        #         'precision': Precision,
        #         'recall': Recall,
        #         'img_mse': Image_Mse,
        #         'rand_img_mse': Rand_Image_Mse,
        #         'attack_total_time': attack_total_time,
        #     }