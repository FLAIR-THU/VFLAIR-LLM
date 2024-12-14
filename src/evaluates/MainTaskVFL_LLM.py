import gc
import sys, os
import re
sys.path.append(os.pardir)
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import time
import copy
import collections
from accelerate import init_empty_weights
import pickle

from sklearn.metrics import roc_auc_score, matthews_corrcoef
import scipy.stats as stats
import torch.nn as nn
import torch
from nltk.translate.bleu_score import sentence_bleu

import inspect
from typing import List, Optional, Tuple, Union, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import namedtuple
import warnings
from typing import List, Optional, Tuple, Union

from transformers import Qwen2ForCausalLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers.generation import GenerationMixin
from transformers.models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)
# from load.LoadModels import QuestionAnsweringModelOutput, SequenceClassifierOutput, CausalLMOutputWithPast


# from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res, multiclass_auc
from utils.communication_protocol_funcs import get_size_of,get_total_size
from party.party_utils import get_model_folder
# from evaluates.attacks.attack_api import apply_attack
from utils import timer
from utils.constants import *
import utils.constants as shared_var
from utils.marvell_functions import KL_gradient_perturb
from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.communication_protocol_funcs import compress_pred, Cache, ins_weight
from utils.squad_utils import normalize_answer, _get_best_indexes, get_tokens, compute_exact, compute_f1

from loguru import logger
from evaluates.defenses.defense_api import apply_defense
from evaluates.defenses.defense_functions import *
from evaluates.attacks.attack_api import AttackerLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from config import vfl_basic_config, is_test

from party.LocalCommunication import LocalCommunication
from framework.client.DistributedCommunication import convert_msg_to_pred, convert_msg_to_tensor
import warnings
warnings.filterwarnings("ignore")

from models.llm_models.minigpt4.minigpt4 import MiniGPT4Tail
from evaluates.vqa_evaluator import VQAEval
from evaluates.gms8k_evaluator import GMS8KEval


torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40, 'diabetes': 0.69, \
                'nuswide': 0.88, 'breast_cancer_diagnose': 0.88, 'adult_income': 0.84, 'cora': 0.72, \
                'avazu': 0.83, 'criteo': 0.74, 'nursery': 0.99, 'credit': 0.82, 'news20': 0.8, \
                'cola_public': 0.8,
                'SST-2': 0.9}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


def create_main_task(global_model_type: GenerationMixin):
    print('###### inherited:', global_model_type,' ######')

    class MainTaskVFL_LLM(global_model_type, nn.Module):  # GenerationMixin object,
        def __init__(self, args, job_id=None):
            # if args.model_architect == 'MM':
            #     super(global_model_type, self).__init__(llm_tokenizer = args.tokenizer)
            # else:
            super(global_model_type, self).__init__(args.config)

            self.job_id = job_id
            self.args = args
            self.config = args.config  # origin model config
            
            # with init_empty_weights():
            #     super().__init__(self.config)  # init CauselLM

            ## VFL configs ##
            self.k = args.k # party number

            ## Device Configs ##
            self.current_device = args.device
            self._device = torch.device(self.current_device)

            ## Task Configs ##
            self.dataset_name = args.dataset
            self.num_classes = args.num_classes
            self.exp_res_dir = args.exp_res_dir

            ## Training Configs ##
            self.epochs = args.main_epochs
            self.lr = args.main_lr
            self.batch_size = args.batch_size
            self.early_stop_threshold = args.early_stop_threshold # Early Stop
            self.Q = args.Q  # FedBCD
            self.is_first_forward_iter = 1

            ## Model Configs ##
            self.models_dict = args.model_list
            
            ## Generation Configs ##
            if self.generation_config != None:
                self.generation_config.update(**args.generation_config_dict)
                print('--> final self.generation_config:',args.generation_config)
                self.generation_config_dict = self.generation_config.to_dict() #args.generation_config_dict # type=Dict
                # self.generation_config_dict.update((k, self.generation_config[k]) for k in self.generation_config.keys())
                print('--> final self.generation_config_dict:',args.generation_config_dict)

            self.exp_res_path = args.exp_res_path

            ###### init Parties ######
            self.parties = args.parties


            ### Results 
            self.real_generation_result = None
            self.loss = None
            self.train_acc = None
            self.flag = 1
            self.stopping_iter = 0
            self.stopping_time = 0.0
            self.stopping_commu_cost = 0
            self.communication_cost = 0
            self.training_time = 0
            self.train_party_time = [0 for i in range(self.k)]
            self.inference_party_time = [0 for i in range(self.k)]

            self.final_epoch = 0
            self.current_epoch = 0
            self.current_step = 0

            self.parties_data = None
            self.gt_one_hot_label = None
            self.clean_one_hot_label = None

            self.pred_list = []
            self.pred_list_clone = []
            self.pred_gradients_list = []
            self.pred_gradients_list_clone = []

            self.local_pred_list = []
            self.local_pred_list_clone = []
            self.local_pred_gradients_list = []
            self.local_pred_gradients_list_clone = []
            
            self.global_pred_dict = {}

            
            # some state of VFL throughout training process
            self.first_epoch_state = None
            self.middle_epoch_state = None
            self.final_state = {}
            self.sample_state = {}
            self.test_sample_state_list = []

            self.num_update_per_batch = args.num_update_per_batch
            self.num_batch_per_workset = args.Q  # args.num_batch_per_workset
            self.max_staleness = self.num_update_per_batch * self.num_batch_per_workset
            
            # self.e2e_model = None  # type:E2EModel
            # self._init_e2e_model()

        @property
        def device(self):
            return self._device

        @device.setter
        def device(self, device):
            if device == self._device:
                print("\nYou are already watching device ", device)
            else:
                self._device = device
                print("\nYou are now watching device ", device)

        def label_to_one_hot(self, target, num_classes=10):
            target = torch.tensor(target)
            target = target.long()
            # print('label_to_one_hot:', target, type(target),type(target[0]))
            try:
                _ = target.size()[1]
                # print("use target itself", target.size())
                onehot_target = target.type(torch.float32).to(self.current_device)
            except:
                target = torch.unsqueeze(target, 1).to(self.current_device)
                # print("use unsqueezed target", target.size(),type(target))
                onehot_target = torch.zeros(target.size(0), num_classes, device=self.current_device)
                onehot_target.scatter_(1, target, 1)
            return onehot_target

        
        def init_communication(self, communication=None):
            if communication is None:
                communication = LocalCommunication(self.args.parties[self.args.k - 1])
            self._communication = communication

        def LR_Decay(self, i_epoch):
            for ik in range(self.k-1):
                self.parties[ik].LR_decay(i_epoch)
            self._communication.send_global_lr_decay(i_epoch)

        def apply_defense_on_pred_transmission(self, pred_detach):
            ########### Defense applied on pred transmit ###########
            if self.args.apply_defense == True and self.is_first_forward_iter== 1:
                if (self.args.apply_dp == True and 'pred' in self.args.dp_add_position) or\
                    (self.args.apply_gs == True and 'pred' in self.args.gs_add_position):
                    # print('before pred_detach:',pred_detach.shape)
                    pred_detach = torch.stack(self.launch_defense(pred_detach, "pred"))
                    # print('after pred_detach:',pred_detach.shape)
            return pred_detach
        
        def apply_defense_on_grad_transmission(self, grad):
            ########### Defense applied on grad transmit ###########
            # print('apply_defense_on_grad_transmission')
            # print('self.args.apply_dp:',self.args.apply_dp, self.args.dp_add_position)
            # print('self.args.apply_gs:',self.args.apply_gs)
            if self.args.apply_defense == True:
                if (self.args.apply_dp == True and 'grad' in self.args.dp_add_position) or \
                    (self.args.apply_gs == True and 'grad' in self.args.gs_add_position):
                    grad = self.launch_defense(grad, "gradients")
                    # print('after grad:',type(grad),grad.shape) # torch.size bs,12,768 intermediate
            return grad


        # def apply_communication_protocol_on_transmission(self, pred_detach):
        #     ########### communication_protocols ###########
        #     if self.args.communication_protocol in ['Quantization', 'Topk']:
        #         pred_detach = compress_pred(self.args, pred_detach, self.parties[ik].local_gradient, \
        #                                     self.current_epoch, self.current_step).to(self.args.device)
        #     return pred_detach

        def pred_transmit(self, use_cache=None, count_time=False):
            '''
            Active party gets pred from passive parties
            '''
            ik = self.current_client_id # for ik in range(self.k - 1):
            result_dict = self.parties[ik].give_pred(use_cache=use_cache)  # use_cache=use_cache
            
            if not ('encoder_outputs' in result_dict.keys()): #self.args.model_config.is_encoder_decoder:
                pred_detach = result_dict['inputs_embeds']
                # Defense
                if self.args.apply_defense:
                    if (ik in self.args.defense_configs['party']):
                        pred_detach = self.apply_defense_on_pred_transmission(pred_detach)
                
                # Communication Process
                # pred_detach = self.apply_communication_protocol_on_transmission(pred_detach)
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                result_dict['inputs_embeds'] = pred_clone

            else: # inference process of encoder-decoder LLMs
                pred_clone = result_dict['encoder_outputs']['last_hidden_state']
            
            passive_pred = result_dict
            get_total_size(result_dict)
            
            self.parties[self.k - 1].receive_pred(result_dict, ik)
            self.parties[ik].update_local_pred(pred_clone)
        
            # print(f'#pred_transmit party{ik}:',result_dict.keys())
            return result_dict

        @timer()
        def global_pred_transmit(self, passive_pred, use_cache=None, count_time=False):
            # print(f'#global_pred_transmit party{self.current_client_id}:',passive_pred.keys())
            
            global_pred = self._communication.send_pred_message(passive_pred, self.current_client_id,\
                self.parse_pred_message_result,use_cache=use_cache)  
            
            get_total_size(global_pred)
            return global_pred
        
        def parse_pred_message_result(self, result):
            if self.args.task_type == 'SequenceClassification':
                logits = torch.Tensor(result['logits'])
                if result['requires_grad']:
                    logits.requires_grad_()
                return SequenceClassifierOutput(
                    logits=logits,
                )
            elif self.args.task_type == 'CausalLM':
                if self.args.vfl_model_slice_num == 3:
                    return convert_msg_to_pred(result)
                logits = convert_msg_to_tensor(result)
                return CausalLMOutputWithPast(
                    logits=logits,
                )
            elif self.args.task_type == 'QuestionAnswering':
                start_logits = torch.Tensor(result['start_logits'])
                end_logits = torch.Tensor(result['end_logits'])
                if result['requires_grad']:
                    start_logits.requires_grad_()
                    end_logits.requires_grad_()
                return QuestionAnsweringModelOutput(
                    loss=None,
                    start_logits=start_logits.to(self.args.device),
                    end_logits=end_logits.to(self.args.device),
                    hidden_states=None,
                    attentions=None,
                )
            elif self.args.task_type == 'DevLLMInference':
                return convert_msg_to_pred(result)
            else:
                assert 1 > 2, 'Task type no supported'

        def local_gradient_transmit(self, count_time='train'):
            '''
            active party --[local gradient]--> passive party
            '''
            # print('#local_gradient_transmit client:',self.current_client_id)
            if self.parties[self.current_client_id].local_model_optimizer != None:
                passive_local_gradient = self._communication.send_cal_passive_local_gradient_message(self.current_client_id)
                if not isinstance(passive_local_gradient, torch.Tensor):
                    passive_local_gradient = torch.Tensor(passive_local_gradient).to(self.args.device)
                
                # Direct alter on gradients
                if self.args.apply_defense:
                    if (1 in self.args.defense_configs['party']):
                        passive_local_gradient = self.apply_defense_on_grad_transmission(passive_local_gradient)
                
                self.parties[self.current_client_id].local_gradient = passive_local_gradient

        def global_gradient_transmit(self, final_pred, count_time='train'):
            '''
            passive party --[global gradient]--> active party
            '''
            # print('#global_gradient_transmit client:',self.current_client_id)

            ik = self.current_client_id
            self.parties[ik].cal_loss(final_pred)

            if self.args.vfl_model_slice_num == 2:
                global_gradient = self.parties[ik].cal_global_gradient_2slice(self.parties[ik].global_loss, final_pred)
            else:
                global_gradient = self.parties[ik].cal_global_gradient_3slice(self.parties[ik].global_loss, self.parties[ik].active_intermediate)

            #### Defense ####
            # Direct alter on gradients
            if self.args.apply_defense:
                if (ik in self.args.defense_configs['party']):
                    if (not self.args.apply_mid) and (not self.args.apply_adversarial):
                        global_gradient = self.apply_defense_on_grad_transmission(global_gradient)

            # Update_loss_with_defense after gradient calculation -- for ad defense at model tail
            self.parties[ik].update_loss_with_defense()
            #### Defense ####
  
            self._communication.send_global_loss_and_gradients(global_gradient, self.current_client_id) 
            self.communication_cost += get_size_of(global_gradient)
        
        def predict(self):
            # passive party dataloader list
            data_loader_list = [self.parties[ik].test_loader for ik in range(self.args.k - 1)]

            exact_score_list = []
            f1_list = []

            nbest_list = []
            gold_ans_list = []

            target_word_list = []
            predict_word_list = []

            target_answer_list = []
            predict_answer_list = []

            predict_label_list = []
            actual_label_list = []
            self._loss = 0
            _batch_cnt = 0
            total_sample_cnt = 0
            with torch.no_grad():
                for parties_data in tqdm(zip(*data_loader_list), desc="inference process"):
                    _batch_cnt += 1
                    _parties_data = []
                    for party_id in range(len(parties_data)):  # parties_data[party_id]: list of bs
                        batch_input_dicts = []
                        batch_label = []
                        for bs_id in range(len(parties_data[party_id])):
                            batch_input_dicts.append(parties_data[party_id][bs_id][0]) # Input Dict
                            batch_label.append(parties_data[party_id][bs_id][1]) # Label
                        _parties_data.append([batch_input_dicts, batch_label])
                    self.parties_data = _parties_data # party_number * [batch_input_dicts, batch_label]
                    
                    self.gt_one_hot_label = [] # party_number * party_label
                    for party_id in range(len(parties_data)):
                        if self.args.model_architect == 'CLS' and self.args.num_classes > 1:  # classification
                            self.gt_one_hot_label.append(self.label_to_one_hot(self.parties_data[party_id][1], self.args.num_classes))
                        elif self.args.model_architect == 'TQA':
                            self.gt_one_hot_label.append(list(self.parties_data[party_id][1]))
                        else:
                            self.gt_one_hot_label.append(self.parties_data[party_id][1])
                    
                    data_input_list = [] # party_number * data_inputs(dict)
                    for party_id in range(len(parties_data)):
                        data_inputs = {}
                        for key_name in self.parties_data[party_id][0][0].keys():
                            if isinstance(self.parties_data[party_id][0][0][key_name], torch.Tensor):
                                data_inputs[key_name] = torch.stack( [self.parties_data[party_id][0][i][key_name] for i in range(len(self.parties_data[0][0]))] )
                            else:
                                data_inputs[key_name] =  [self.parties_data[party_id][0][i][key_name] for i in range(len(self.parties_data[0][0]))]
                        
                        if 'input_ids' in data_inputs.keys() and data_inputs['input_ids']!=None:
                            self.seq_length = data_inputs['input_ids'].shape[-1]
                        else:
                            self.seq_length = 0
                        data_input_list.append(data_inputs)
                    
                    if self.args.model_architect == 'CLS':  # task_type == "SequenceClassification":  # and self.args.num_classes > 1: # classification
                        for party_id in range(self.args.k-1):
                            party_global_output = self.forward(**data_input_list[party_id])
                            batch_predict_label, batch_actual_label, sample_cnt = self.generate_result(party_global_output,
                                                                                            self.gt_one_hot_label[party_id])
                            predict_label_list.extend(batch_predict_label)
                            actual_label_list.extend(batch_actual_label)
                            if sample_cnt is not None:
                                total_sample_cnt += sample_cnt
                    
                    elif self.args.model_architect == 'TQA':  # task_type == "QuestionAnswering":
                        for party_id in range(self.args.k-1):
                            party_global_output = self.forward(**data_input_list[party_id])
                            feature_list = data_input_list[party_id]['feature']
                            batch_nbest, batch_gold_ans, sample_cnt = self.generate_result(party_global_output, self.gt_one_hot_label[party_id], feature_list)
                            nbest_list.extend(batch_nbest)
                            gold_ans_list.extend(batch_gold_ans)
                            if sample_cnt is not None:
                                total_sample_cnt += sample_cnt
                    
                    elif self.args.model_architect=='CLM': #task_type == "CausalLM":
                        for party_id in range(self.args.k-1):
                            if not (self.args.max_new_tokens==1):
                                self.set_is_first_forward_epoch(1)
                                party_generation_output = self.generate(**data_input_list[party_id], \
                                        generation_config = self.generation_config)
                            else:  # next token prediction
                                party_generation_output = self.forward(**data_input_list[party_id])
                            
                            self._clear_past_key_values()

                            batch_target_word, batch_predict_word, sample_cnt = self.generate_result(party_generation_output, self.gt_one_hot_label[party_id])
                            target_word_list.extend(batch_target_word)
                            predict_word_list.extend(batch_predict_word)
                            if sample_cnt is not None:
                                total_sample_cnt += sample_cnt
                        # else:
                        #     global_output = self.forward(**data_inputs)
                        #     batch_predict_label, batch_actual_label, sample_cnt = self.generate_result(global_output, self.gt_one_hot_label)
                        #     predict_label_list.extend(batch_predict_label)
                        #     actual_label_list.extend(batch_actual_label)
                        #     if sample_cnt is not None:
                        #         total_sample_cnt += sample_cnt
                    
                    elif self.args.model_architect=='MM':
                        for party_id in range(self.args.k-1):
                            if not (self.args.max_new_tokens==1):
                                self.set_is_first_forward_epoch(1)
                                if self.args.generation_method == 'chat':
                                    party_generation_output = self.mm_chat(samples = data_input_list[party_id], # 
                                                                        **self.generation_config_dict
                                                                        )
                                elif self.args.generation_method == 'generate':
                                    party_generation_output = self.mm_generate(samples = data_input_list[party_id], # 
                                                                        generation_config = self.generation_config,
                                                                        **self.generation_config_dict)
                                else:
                                    assert 1>2,f'Generation Method {self.args.generation_method} not supported'
                            else:  # next token prediction
                                party_generation_output = self.forward(samples = data_input_list[party_id], labels = self.gt_one_hot_label[party_id])
                            self._clear_past_key_values()

                            batch_target_answer, batch_predict_answer, sample_cnt = self.generate_result(party_generation_output, self.gt_one_hot_label[party_id])
                            target_answer_list.extend(batch_target_answer)
                            predict_answer_list.extend(batch_predict_answer)

                            if sample_cnt is not None:
                                total_sample_cnt += sample_cnt
                        
                    else:
                        assert 1 > 2, 'Task type not supported'

                    if self.args.need_test_sample_states and (len(self.test_sample_state_list) < 100):
                        self.sample_state.update(self.save_state(False))
                        self.test_sample_state_list.append(self.sample_state)
                        self.sample_state = {}

                    del parties_data
            
            if self._loss:
                self._loss = self._loss / _batch_cnt
            
            if self.args.model_architect == 'CLS':  # and self.args.num_classes > 1: # classification
                return predict_label_list, actual_label_list, total_sample_cnt
            elif self.args.model_architect == 'TQA':
                return nbest_list, gold_ans_list, total_sample_cnt
            elif self.args.model_architect=='CLM':
                if self.args.task_type == "CausalLM":
                    return predict_word_list, target_word_list, total_sample_cnt
                else:
                    return predict_label_list, actual_label_list, total_sample_cnt
            elif self.args.model_architect=='MM':
                return predict_answer_list, target_answer_list, total_sample_cnt
            else:
                assert 1 > 2, 'Task type not supported'

        def generate_assessment(self, predict_list, label_list):
            if self.args.model_architect == 'TQA':
                nbest_list = predict_list
                gold_ans_list = label_list

                best_non_null_entry = None
                exact_score_list = []
                f1_list = []
                for nbest, gold_ans in zip(nbest_list, gold_ans_list):  # iterate through each sample
                    exact_score = 0
                    f1 = 0
                    best_non_null_entry = None
                    if self.args.metric_type == "best_pred":
                        for entry in nbest:
                            # total_scores.append(entry.start_logit + entry.end_logit)
                            if not best_non_null_entry:
                                if entry.text:
                                    best_non_null_entry = entry
                        pred_ans_text = best_non_null_entry.text if (best_non_null_entry != None) else ""
                        exact_score = max(compute_exact(a, pred_ans_text) for a in gold_ans)
                        f1 = max(compute_f1(a, pred_ans_text) for a in gold_ans)
                        exact_score_list.append(exact_score)
                        f1_list.append(f1)
                    elif self.args.metric_type == "n_best":
                        for entry in nbest:
                            total_scores.append(entry.start_logit + entry.end_logit)
                            if not best_non_null_entry:
                                if entry.text:
                                    best_non_null_entry = entry
                            pred_ans_text = entry.text
                            exact_score = max(exact_score, max(compute_exact(a, pred_ans_text) for a in gold_ans))
                            f1 = max(f1, max(compute_f1(a, pred_ans_text) for a in gold_ans))
                        exact_score_list.append(exact_score)
                        f1_list.append(f1)
                    else:
                        assert 1 > 2, f"{self.args.metric_type} not provided!"

                exact_score = np.mean(exact_score_list)
                f1 = np.mean(f1_list)
                return {'exact_score': exact_score, 'f1': f1}

            elif self.args.model_architect == 'CLM':
                if self.args.task_type == "CausalLM":
                    predict_word_list = predict_list # bs, seq_len, vocab_size
                    target_word_list = label_list # bs, seq_len

                    if len(target_word_list[0].shape)>0: # long text generation, not next token prediction   
                        # print('target_word_list:',len(target_word_list),target_word_list[0].shape)
                        # print('predict_word_list:',len(predict_word_list),predict_word_list[0].shape)
                        
                        if self.args.dataset == 'GMS8K' or self.args.dataset == 'GMS8K-test':
                            self.evaluator = GMS8KEval(self.args)
                            acc = self.evaluator.evaluate(predict_word_list,target_word_list )
                        
                        elif self.args.dataset=='MATH':
                            # print('predict_word_list:',type(predict_word_list),len(predict_word_list),predict_word_list[0].shape)
                            
                            def wash(token_id_list, washed_ids):
                                washed_token_id_list = []
                                for token_ids in token_id_list:
                                    token_ids = list(token_ids)
                                    for washed_id in washed_ids:
                                        while washed_id in token_ids:
                                            token_ids.remove(washed_id)
                                    
                                    washed_token_id_list.append(torch.tensor(token_ids) )
                                return washed_token_id_list

                            washed_ids = [self.args.tokenizer.pad_token_id, self.args.tokenizer.eos_token_id, self.args.tokenizer.bos_token_id]
                            predict_word_list = wash(predict_word_list,washed_ids )
                            target_word_list = wash(target_word_list,washed_ids )

                            predict_word_list = [
                                self.args.tokenizer.decode(_ids)
                                for _ids in list(predict_word_list)]

                            target_word_list = [
                                self.args.tokenizer.decode(_ids)
                                for _ids in list(target_word_list)]
                            
                            def is_equiv(str1, str2, verbose=False):
                                if str1 is None and str2 is None:
                                    print("WARNING: Both None")
                                    return True
                                if str1 is None or str2 is None:
                                    return False

                                try:
                                    ss1 = strip_string(str1)
                                    ss2 = strip_string(str2)
                                    #pdb.set_trace()
                                    if verbose:
                                        print(ss1, ss2)
                                    return ss1 == ss2
                                except Exception:
                                    return str1 == str2
                            
                            def process_results(completion, answer): # doc
                                split_ans = completion.split('The answer is: ')
                                if len(split_ans) > 1:
                                    ans = split_ans[-1]
                                    extract_ans_temp = ans.split('.\n')[0]
                                    extract_ans_temp = extract_ans_temp.strip()
                                    if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
                                        extract_ans = extract_ans_temp[0:-1]
                                    else:
                                        extract_ans = extract_ans_temp
                                    extract_ans = extract_ans.strip()
                                    
                                    # print('extract_ans:',extract_ans)
                                    # print('answer:',answer)

                                    if is_equiv(extract_ans, answer):
                                        return True
                                    else:
                                        return False
                                else:
                                    # temp = {'question': doc, 'output': completion, 'answer': answer}
                                    # invalid_outputs.append(temp)
                                    return False

                            results = []
                            for i in range(len(target_word_list)):
                                res = process_results(predict_word_list[i],target_word_list[i])
                                # print('-'*100)
                                # print('PRED:',predict_word_list[i])
                                # print('GOLD:',target_word_list[i])
                                # print('SCORE:',res)
                                results.append(res)
                            acc = sum(results) / len(results)

                        else:
                            
                            def calculate_token_precision_recall(reference_ids, candidate_ids):
                                reference_ids = reference_ids.tolist()
                                candidate_ids = candidate_ids.tolist()

                                def wash(ids, target_token_id):
                                    while(target_token_id in ids):
                                        ids.remove(target_token_id)
                                    return ids
                                
                                reference_ids = wash(reference_ids, self.args.tokenizer.pad_token_id)
                                reference_ids = wash(reference_ids, self.args.tokenizer.eos_token_id)

                                reference_tokens = [self.args.tokenizer.convert_ids_to_tokens(reference_ids)]
                                candidate_tokens = self.args.tokenizer.convert_ids_to_tokens(candidate_ids)
                                

                                score = sentence_bleu(reference_tokens, candidate_tokens)

                                print('Reference_tokens:',reference_tokens)
                                print('-'*25)
                                print('Candidate_tokens',candidate_tokens)
                                print('Score:',score)
                                print('='*50)
                                assert 1>2
                                return score
                            

                            score = 0
                            for i in range(len(target_word_list)):
                                _score = calculate_token_precision_recall(target_word_list[i], predict_word_list[i])
                                score += _score
                            score = score/len(target_word_list)
                            acc = score
                                
                    else:
                        if self.args.metric_type == "best_pred":
                            suc_cnt = 0
                            for i in range(len(target_word_list)):
                                if target_word_list[i] == predict_word_list[i]:
                                    suc_cnt += 1
                            acc = suc_cnt / float(len(target_word_list))
                        elif self.args.metric_type == "n_best":
                            suc_cnt = 0
                            for i in range(len(target_word_list)):
                                if target_word_list[i] in predict_word_list[i]:
                                    suc_cnt += 1
                            acc = suc_cnt / float(len(target_word_list))  # ACC
                        else:
                            assert 1 > 2, 'metric type not supported'
                        
                    return {'acc':acc}
                else:
                    predict_labels = predict_list
                    actual_labels = label_list

                    suc_cnt = torch.sum(torch.tensor(predict_labels) == \
                                        torch.tensor(actual_labels)).item()
                    acc = suc_cnt / torch.tensor(predict_labels).shape[0]  # ACC
                    mcc = matthews_corrcoef(np.array(predict_labels), np.array(actual_labels))  # MCC

                    return {'acc':acc, 'mcc':mcc}

            elif self.args.model_architect == 'MM':
                predict_word_list = predict_list
                target_word_list = label_list
                # print('--- generate_assessment ---')
                # print('predict_word_list:',type(predict_word_list),len(predict_word_list),predict_word_list)
                # print('target_word_list:',type(target_word_list),len(target_word_list),target_word_list)
               
                total_accuracy = 0
                num = 0
                # Entry = collections.namedtuple('Entry', ['text', 'bbox'])
                for idx in range(len(predict_word_list)):
                    answer = predict_word_list[idx] #item['gt_answers']
                    gt_answers = target_word_list[idx] #item['answer']
                    
                    self.evaluator = VQAEval()
                    accuracy = self.evaluator.evaluate_vqa_human(answer, gt_answers)
                   
                    # item['accuracy'] = accuracy

                    total_accuracy += accuracy
                    num += 1
                    # print(f'{idx} gt_answers:',gt_answers)
                    # print(f'{idx} answer:',answer)
                    # print(f'{idx} accuracy:',accuracy)

                average_accuracy = total_accuracy / num
                print('-- average_accuracy:',average_accuracy)
                return {'acc':average_accuracy}
               
            elif self.args.model_architect == 'CLS':
                predict_labels = predict_list
                actual_labels = label_list
                if self.num_classes == 1:
                    mse = torch.mean(
                        (torch.tensor(predict_labels) - torch.tensor(actual_labels)) ** 2).item()
                    pearson_corr = stats.pearsonr(torch.tensor(predict_labels), torch.tensor(actual_labels))[0]
                    spearmanr_corr = stats.spearmanr(torch.tensor(predict_labels), torch.tensor(actual_labels))[0]
                    return {'mse': mse, 'pearson_corr': pearson_corr, 'spearmanr_corr': spearmanr_corr}
                else:
                    suc_cnt = torch.sum(torch.tensor(predict_labels) == \
                                        torch.tensor(actual_labels)).item()
                    acc = suc_cnt / torch.tensor(predict_labels).shape[0]  # ACC
                    mcc = matthews_corrcoef(np.array(predict_labels), np.array(actual_labels))  # MCC

                    return {'acc': acc, 'mcc': mcc}

        def generate_result(self, model_output, gt_one_hot_label, feature_list=None):
            # raw_model_output --> standard prediction result
            test_preds = []
            test_targets = []
            test_predict_labels = []
            test_actual_labels = []
            target_word_list = []
            predict_word_list = []
            suc_cnt = 0
            sample_cnt = 0

            if self.args.model_architect == 'CLS':  # task_type == "SequenceClassification":
                if self.args.num_classes == 1:  # regression
                    predict_label = model_output.logits.detach().cpu()
                    actual_label = gt_one_hot_label.detach().cpu()

                    predict_label = torch.tensor([_.item() for _ in predict_label])
                    actual_label = torch.tensor([_.item() for _ in actual_label])

                    sample_cnt = predict_label.shape[0]

                    return list(predict_label), list(actual_label), sample_cnt
                else:  # Classification
                    predict_label = torch.argmax(model_output.logits, dim=-1).detach().cpu()
                    actual_label = torch.argmax(gt_one_hot_label, dim=-1).detach().cpu()

                    sample_cnt = predict_label.shape[0]
                    suc_cnt += torch.sum(predict_label == actual_label).item()
                    return list(predict_label), list(actual_label), sample_cnt

            elif self.args.model_architect == 'MM':  # .task_type == "CausalLM":
                def _decode_text(result_ids, tokenizer):
                    result_text = []
                    for result in result_ids:
                        result = result[result != 0]
                        if result[0] == tokenizer.bos_id:
                            result = result[1:]
                        if result[-1] == tokenizer.eos_id:
                            result = result[:-1]
                        result_text.append(tokenizer.decode(result).strip())
                    return result_text
                
                predict_label_list = _decode_text(model_output,self.args.tokenizer)
                # print('predict_label_list:',model_output.shape,predict_label_list)
                target_label_list = list(gt_one_hot_label)
                # print('target_label_list:',len(target_label_list),target_label_list)

                return target_label_list, predict_label_list, len(predict_label_list) 

            elif self.args.model_architect == 'CLM':  # .task_type == "CausalLM":
                if self.args.task_type == "CausalLM":  # dataset == "Lambada":
                    if isinstance(model_output, torch.Tensor):  # generation -- generated token ids
                        predict_label_list = model_output[:,self.seq_length:] # [bs, max_new_tokens]
                        target_label_list = list(gt_one_hot_label)

                    else:  # forward -- raw model output next token logits
                        # print('model_output:',type(model_output),model_output.keys())
                        generated_token_logits = model_output.logits[:,-1,:]
                        predict_label_list = torch.argmax(generated_token_logits, dim=-1) 
                        target_label_list = list(gt_one_hot_label)
                    
                    self.real_generation_result = predict_label_list

                    return target_label_list, predict_label_list, len(predict_label_list) 

                elif self.args.task_type == "SequenceClassification":
                    # print('gt_one_hot_label:',gt_one_hot_label)
                    # print(self.args.tokenizer.decode(gt_one_hot_label))

                    # choice_dict = {self.args.tokenizer("A").input_ids[0]:0,
                    #     self.args.tokenizer("B").input_ids[0]:1, 
                    #     self.args.tokenizer("C").input_ids[0]:2, 
                    #     self.args.tokenizer("D").input_ids[0]:3}
                    
                    choice_dict = {self.args.tokenizer.convert_tokens_to_ids("A"):0,
                        self.args.tokenizer.convert_tokens_to_ids("B"):1, 
                        self.args.tokenizer.convert_tokens_to_ids("C"):2, 
                        self.args.tokenizer.convert_tokens_to_ids("D"):3}

                    target_label_list = []
                    for row in gt_one_hot_label:
                        tokens = choice_dict[row.item()]
                        target_label_list.append(tokens)
                    
                    # print('target_label_list:',target_label_list)

                    logits = model_output.logits[:,-1,:]
                    probs = (
                            torch.stack(
                                [
                                    logits[:,self.args.tokenizer("A").input_ids[0]],
                                    logits[:,self.args.tokenizer("B").input_ids[0]],
                                    logits[:,self.args.tokenizer("C").input_ids[0]],
                                    logits[:,self.args.tokenizer("D").input_ids[0]],
                                ]
                            ,dim=1)
                        .detach()
                        .cpu()
                    )
                    # # forward -- raw model output
                    # generated_token_logits = model_output.logits[:,-1,:]
                    # probs = []
                    # for choice_class in range(self.args.num_classes):
                    #     choice_id = self.args.tokenizer.convert_tokens_to_ids(self.args.label_dict[choice_class])
                    #     probs.append( generated_token_logits[:, choice_id] ) # [bs, 1]
                    # probs = torch.stack(probs,dim = -1) # [bs, num_choice]



                    predict_label_list = torch.argmax(probs, dim=-1)  # [bs]

                    # predict_label_list = [self.args.label_dict[pred_class.item()] for pred_class in predict_label_list]
                    # predict_label_list = [self.args.tokenizer.convert_tokens_to_ids(pred_token)\
                    #          for pred_token in predict_label_list]
                    # print('predict_label_list:',predict_label_list)

                    return target_label_list, predict_label_list, len(predict_label_list) 

            elif self.args.model_architect=='TQA': #.task_type == "QuestionAnswering":
                # print('feature_list:',type(feature_list),len(feature_list),type(feature_list[0]))
                start_logits = model_output.start_logits # bs, 512
                end_logits = model_output.end_logits # bs, 512
                sample_cnt = start_logits.shape[0] # bs

                n_best_size = self.args.n_best_size
                start_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in start_logits]
                end_indexes = [_get_best_indexes(_logits, n_best_size) for _logits in end_logits]
                # start_indexes: list bs * n_nest_size [nbest start index]

                exact_score_list = []
                f1_list = []
                batch_nbest_list = []
                batch_gold_ans_list = []
                # print('feature_list:',type(feature_list),len(feature_list))
                for i in range(start_logits.shape[0]):  # for each sample in this batch
                    ############ Gold ################
                    # feature = parties_data[0][0][i]['feature']  # print('parties_data[0][4]:',type(parties_data[0][4]),'feature:',type(feature))
                    # feature_tokens = [_token for _token in feature["tokens"]]  # [_token[0] for _token in feature["tokens"]]
                    feature = feature_list[i]

                    gold_start_indexs, gold_end_indexs = gt_one_hot_label[i]  # the i'th sample in a batch
                    # train: int // test: torch.tensor[int1, int2..]
                    if isinstance(gold_start_indexs, int):
                        gold_start_indexs = [gold_start_indexs]
                        gold_end_indexs = [gold_end_indexs]

                    # if len(gold_start_indexs.shape) == 0:
                    #     gold_start_indexs = gold_start_indexs.unsqueeze(0)
                    # if len(gold_end_indexs.shape) == 0:
                    #     gold_end_indexs = gold_end_indexs.unsqueeze(0)

                    gold_ans = []  # gold answers for this sample
                    for _i in range(len(gold_start_indexs)):
                        gold_start_index = int(gold_start_indexs[_i])
                        gold_end_index = int(gold_end_indexs[_i])

                        gold_ans_text = list(range(gold_start_index, gold_end_index + 1))
                        gold_ans.append(gold_ans_text)

                    batch_gold_ans_list.append(gold_ans)

                    ############ Pred ################
                    _start_logits = start_logits[i]
                    _end_logits = end_logits[i]
                    _start_indexes = start_indexes[i]  # [nbest start index] list  n_best_size
                    _end_indexes = end_indexes[i]  # [nbest end index] list  n_best_size

                    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                        "PrelimPrediction",
                        ["start_index", "end_index", "start_logit", "end_logit"])
                    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                        "NbestPrediction", ["text", "start_logit", "end_logit"])
                    # throw out all invalid predictions.
                    prelim_predictions = []
                    for start_index in _start_indexes:
                        for end_index in _end_indexes:
                            # We could hypothetically create invalid predictions, e.g., predict
                            # that the start of the span is in the question. We throw out all
                            # invalid predictions.
                            if start_index >= feature['len_tokens']:  # len(feature["tokens"]):
                                continue
                            if end_index >= feature['len_tokens']:  # len(feature["tokens"]):
                                continue
                            if start_index not in feature["token_to_orig_map"]:
                                continue
                            if end_index not in feature["token_to_orig_map"]:
                                continue
                            if not feature["token_is_max_context"].get(start_index, False):
                                continue
                            if end_index < start_index:
                                continue
                            length = end_index - start_index + 1
                            if length > self.args.max_answer_length:
                                continue

                            prelim_predictions.append(
                                _PrelimPrediction(
                                    start_index=start_index,
                                    end_index=end_index,
                                    start_logit=_start_logits[start_index],
                                    end_logit=_end_logits[end_index]))

                    # Iterate through Sorted Predictions
                    prelim_predictions = sorted(
                        prelim_predictions,
                        key=lambda x: (x.start_logit + x.end_logit),
                        reverse=True)  # length=2
                    # print('prelim_predictions:',len(prelim_predictions))

                    exact_score = 0
                    f1 = 0
                    nbest = []  # Get n best prediction text
                    n_best_size = min(n_best_size, len(prelim_predictions))
                    for _id in range(n_best_size):
                        start_index = prelim_predictions[_id].start_index
                        end_index = prelim_predictions[_id].end_index

                        # pred_ans_text = " ".join(feature_tokens[start_index:(end_index + 1)])
                        # pred_ans_text = normalize_answer(pred_ans_text)

                        pred_ans_text = list(range(start_index, end_index + 1))
                        nbest.append(
                            _NbestPrediction(
                                text=pred_ans_text,
                                start_logit=prelim_predictions[_id].start_logit,
                                end_logit=prelim_predictions[_id].end_logit))

                    batch_nbest_list.append(nbest)
                return batch_nbest_list, batch_gold_ans_list, sample_cnt

            else:
                assert 1 > 2, "task_type not supported"

        @timer()
        def _llm_inference(self, **kwargs):
            if self.k > 2:
                raise ValueError('llm_inference only supports k=2')
            start_time = time.time()
            format_kwargs = self._format_forward_kwargs(**kwargs)
            generate_ids = self.generate(format_kwargs.get('input_ids'), max_new_tokens=20)
            generate_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(format_kwargs['input_ids'], generate_ids)
            ]
            resp = self.args.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
            end_time = time.time()
            logger.info(f"Took: {end_time - start_time} s\nGenerated: {resp}")
            return '', ''

        def _format_forward_kwargs(self, **kwargs):
            if not kwargs:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "You are a python programmer, what can you do?"}
                ]
            else:
                messages = kwargs['messages']
            tokenizer = self.args.tokenizer
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt")
            kwargs.update({'input_ids': model_inputs.input_ids.to(self.device),
                           'output_hidden_states': True})
            # logger.debug(f"default inference, kwargs.keys: {kwargs.keys()}")

            base_dict = {'input_ids': None,
                         'attention_mask': None,
                         'position_ids': None,
                         'past_key_values': None,
                         'inputs_embeds': None,
                         'use_cache': False,
                         'output_attentions': None,
                         'output_hidden_states': True,
                         'return_dict': None, }
            for k in base_dict:
                if k in kwargs:
                    base_dict.update({k: kwargs.get(k)})
            return base_dict

        def mm_inference(self):
            self.eval()
            predict_word_list, target_word_list, total_sample_cnt = self.predict()

            result_dict = self.generate_assessment(predict_word_list, target_word_list)
            self.test_acc = result_dict['acc']
            exp_result = f'|test_acc={self.test_acc}'
            # print(exp_result)
            return exp_result, self.test_acc
            # except:
            #     return '',0

        def seq_inference(self):
            # SequenceClassification / Regression
            predict_labels, actual_labels, total_sample_cnt = self.predict()

            # prediction result assessment
            result_dict = self.generate_assessment(predict_labels, actual_labels)
            if self.num_classes == 1:
                self.test_mse = result_dict['mse']
                self.test_pearson_corr = result_dict['pearson_corr']
                self.test_spearmanr_corr = result_dict['spearmanr_corr']

                exp_result = f'|test_mse={self.test_mse}|test_pearson_corr={self.test_pearson_corr}|test_spearmanr_corr={self.test_spearmanr_corr}'
                print(exp_result)
                return exp_result, [self.test_mse, self.test_pearson_corr, self.test_spearmanr_corr]
            else:
                self.test_acc = result_dict['acc']
                self.test_mcc = result_dict['mcc']
                exp_result = f'|test_acc={self.test_acc}|test_mcc={self.test_mcc}'
                print(exp_result)
                return exp_result, self.test_acc

        def causal_lm_inference(self):
            self.eval()
            predict_word_list, target_word_list, total_sample_cnt = self.predict()

            # print('causal_lm_inference target_word_list:',target_word_list)
            # print('causal_lm_inference predict_word_list:',predict_word_list)

            # try:
            result_dict = self.generate_assessment(predict_word_list, target_word_list)
            self.test_acc = result_dict['acc']
            exp_result = f'|test_acc={self.test_acc}'
            # print(exp_result)
            return exp_result, self.test_acc
            # except:
            #     return '',0

        def qa_inference(self):
            # generate all model prediction
            start_time = time.time()
            nbest_list, gold_ans_list, total_sample_cnt = self.predict()
            end_time = time.time()

            start_time = time.time()

            result_dict = self.generate_assessment(nbest_list, gold_ans_list)
            exp_result = '|exact_score={:.4f}|f1={:.4f}'.format(result_dict['exact_score'], result_dict['f1'])

            self.test_acc = result_dict['exact_score']
            print(exp_result)
            return exp_result, self.test_acc

        def forward(self, **kwargs):
            self.parties[self.current_client_id].obtain_local_data(kwargs)

            # passive party do local pred
            passive_party_pred = self.pred_transmit(use_cache=False)

            # passive party inform active party to do global pred
            resp = self.global_pred_transmit(passive_party_pred, use_cache=False)

            if self.args.vfl_model_slice_num > 2:
                p = self.parties[self.current_client_id]
                p._tensor_to_device(resp, p.models[2].device)
                p.active_intermediate = resp['inputs_embeds'] # receive active party's intermediate
                final_output = p.give_final_pred(resp) 
            else:
                final_output = resp
            self.parties[self.current_client_id]._tensor_to_device(final_output,self.device)

            if (self.current_client_id == 0) and self.args.max_new_tokens > 1 and (self.args.need_final_epoch_state or self.args.need_test_sample_states):
                if self.args.need_generation_state:
                    if self.is_first_forward_iter:
                        self.final_state.update(self.save_element('active_predict_list'))
                        self.final_state.update(self.save_element('active_predict_attention_mask_list'))
                        self.sample_state.update(self.save_element('active_predict_list'))
                        self.sample_state.update(self.save_element('active_predict_attention_mask_list'))
                    
                    else:
                        self.final_state['active_predict_list'].append(self.dict_deepcopy(self.parties[1].output_tensors, device = 'cpu'))
                        self.final_state['active_predict_attention_mask_list'].append(self.dict_deepcopy(self.parties[1].output_attention_mask, device = 'cpu'))
                        self.sample_state['active_predict_list'].append(self.dict_deepcopy(self.parties[1].output_tensors, device = 'cpu'))
                        self.sample_state['active_predict_attention_mask_list'].append(self.dict_deepcopy(self.parties[1].output_attention_mask, device = 'cpu'))
                      
                self.set_is_first_forward_epoch(0)

            return final_output

        def mm_chat(self, samples=None, **kwargs):
            # print('--- mm_chat')
            # print('samples length(bs):',len(samples))
            # print(samples)

            aligned_input, generation_config = self.parties[0].local_model.pre_chat(
                sample = samples, context=None, #image = images[0], msgs= msgs, 
                tokenizer = self.args.tokenizer, **kwargs)
            # msgs = aligned_input.pop('msgs')
            
            with torch.inference_mode():
                res, vision_hidden_states = self.mm_generate(
                    **aligned_input,
                    **generation_config
                )
            answer = res[0].unsqueeze(0)
            
            # context = msgs.copy()
            # context.append({"role": "assistant", "content": answer})
            # print('context:',context)

            return answer #, context, generation_config

        def mm_generate(self, samples=None, **kwargs):
            # print('--- mm_generate')
            # print('samples:',samples)
            # print('kwargs:',kwargs.keys())
            
            if 'tokenizer' not in kwargs.keys():
                kwargs['tokenizer'] = self.args.tokenizer

            aligned_input = self.parties[0].local_model.pre_generation(samples=samples, **kwargs)
            # print('after pre_generation aligned_input:',aligned_input.keys())
            return self.generate( **aligned_input)# tokenizer = self.args.tokenizer,

        def transmit_relevant_gradient(self, final_pred_dict):
            '''
            calculate and transmit relevant gradient
            '''
            # passive party -> global gradient -> active party
            for client_id in range(self.k-1):
                self.current_client_id = client_id
                self.global_gradient_transmit(final_pred_dict[client_id], count_time='train')
            
            # active party aggregate received global gradient --> active_party.global_gradient
            self.parties[-1].aggregate_gradients()
            
            # active party -> local gradient -> passive party
            for client_id in range(self.k-1):
                self.current_client_id = client_id
                self.local_gradient_transmit(count_time='train')

        @timer()
        def inference(self, **kwargs):
            # if self.args.attack_only:
            #     self.final_state=self.save_party_data()
            #     return "|attack_only|", -1

            need_save_state = kwargs.get('need_save_state')
            # print('inference need_save_state:',need_save_state)
            if self.args.task_type == "DevLLMInference":
                # LLM推理入口，兼容forward格式入参
                for i in range(20):
                    result = self._llm_inference(**kwargs)
                return result

            # set inference time back to 0
            self.inference_party_time = [0 for i in range(self.k)]

            # print(' ========= Inference ==========')
            for ik in range(self.k - 1):
                # self.parties[ik].prepare_data_loader()
                self.parties[ik].eval()
            self.parties[self.k - 1].eval()

            if self.args.model_architect == 'MM':  # task_type == "MultiModality":
                exp_result, main_task_result = self.mm_inference()
                if need_save_state:
                    self.final_state = self.save_state(False)
                    self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}' + exp_result
                return exp_result, main_task_result


            if self.args.model_architect == 'TQA':  # task_type == "QuestionAnswering":
                exp_result, main_task_result = self.qa_inference()
                if need_save_state:
                    self.final_state = self.save_state(False)
                    self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}' + exp_result
                return exp_result, main_task_result

            if self.args.model_architect == 'CLS':  # task_type == "SequenceClassification":
                # exp_result, self.test_acc =
                exp_result, main_task_result = self.seq_inference()
                if need_save_state:
                    self.final_state = self.save_state(False)
                    self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}' + exp_result
                return exp_result, main_task_result

            if self.args.model_architect == 'CLM':  # task_type == "CausalLM":
                exp_result, main_task_result = self.causal_lm_inference()
                
                if need_save_state:
                    self.final_state.update(self.save_state(False))
                    self.final_state.update(self.save_party_data())
                exp_result = f'|inference_party_time={self.inference_party_time}' + str(exp_result)
                return exp_result, main_task_result

        def train_batch(self, parties_data, batch_label):
            '''
            parties_data : client_number * (batch_input, batch_label)
            batch_label : client_number * batch_label
            '''
            self.set_is_first_forward_epoch(1)

            ############### allocate data ###############
            gt_one_hot_label = batch_label
            
            for ik in range(self.k - 1):
                # allocate data to party ik
                data_inputs = {}
                for key_name in parties_data[ik][0][0].keys():
                    if isinstance(parties_data[ik][0][0][key_name], torch.Tensor):
                        data_inputs[key_name] = torch.stack( [parties_data[ik][0][i][key_name] for i in range(len(parties_data[ik][0]))] )
                    else:
                        data_inputs[key_name] =  [parties_data[ik][0][i][key_name] for i in range(len(parties_data[ik][0]))]
                self.parties[ik].obtain_local_data(data_inputs)
                
                gt_one_hot_label = batch_label[ik]
                self.parties[ik].gt_one_hot_label = gt_one_hot_label
                
                if 'input_ids' in data_inputs.keys() and data_inputs['input_ids']!=None:
                    self.seq_length = data_inputs['input_ids'].shape[-1]
                else:
                    self.seq_length = 0

            ################ normal vertical federated learning ################
            # =================== Forward + Intermediate Commu ===================
            final_pred_dict = {}
            for client_id in range(self.k-1):
                self.current_client_id = client_id
                if self.args.model_architect == 'MM':
                    final_pred = self.forward(samples = data_inputs, labels = gt_one_hot_label)
                else:     
                    final_pred = self.forward(**data_inputs)
                final_pred_dict[client_id] = final_pred
                self._clear_past_key_values()
            
            # =================== Backward + Gradient Commu ===================
            self.transmit_relevant_gradient(final_pred_dict)
            
            # ============= Model Update =============
            self._communication.send_global_backward_message() # active_party.global_backward()
            for ik in range(self.k - 1):
                self.parties[ik].local_backward() # passive_party.local_backward

            ################ normal vertical federated learning ################

            self.set_is_first_forward_epoch(1)

            # Return Loss and Acc
            if self.args.model_architect == 'TQA':  # self.args.task_type == 'QuestionAnswering':
                pred = final_pred  # QuestionAnsweringModelOutput
                loss = self.parties[0].global_loss
                feature_list = data_inputs['feature']
                batch_nbest, batch_gold_ans, sample_cnt = self.generate_result(pred, gt_one_hot_label, feature_list)

                result_dict = self.generate_assessment(batch_nbest, batch_gold_ans)

                exact_score = result_dict['exact_score']
                return loss.item(), exact_score

            elif self.args.model_architect == 'CLS':  # self.args.task_type == 'SequenceClassification':

                if self.args.num_classes == 1:
                    pred = final_pred
                    loss = self.parties[0].global_loss

                    batch_predict_label, batch_actual_label, sample_cnt = self.generate_result(pred, gt_one_hot_label)
                    result_dict = self.generate_assessment(batch_predict_label, batch_actual_label)

                    batch_mse = result_dict['mse']
                    batch_pearson_corr = result_dict['pearson_cor']
                    batch_spearmanr_corr = result_dict['spearmanr_corr']

                    return loss.item(), [batch_mse, batch_pearson_corr, batch_spearmanr_corr]
                else:
                    pred = final_pred
                    loss = self.parties[0].global_loss

                    batch_predict_label, batch_actual_label, sample_cnt = self.generate_result(pred, gt_one_hot_label)
                    result_dict = self.generate_assessment(batch_predict_label, batch_actual_label)

                    batch_train_acc = result_dict['acc']
                    return loss.item(), batch_train_acc

            elif self.args.model_architect=='CLM':  #self.args.task_type == 'CausalLM':
                pred = final_pred
                loss = self.parties[0].global_loss

                batch_train_acc = 0

                return loss.item(), batch_train_acc
            
            elif self.args.model_architect=='MM':  #self.args.task_type == 'CausalLM':
                pred = final_pred
                loss = self.parties[0].global_loss

                batch_train_acc = 0

                return loss.item(), batch_train_acc
        
        def set_is_first_forward_epoch(self, value):
            assert value == 1 or value == 0
            self.is_first_forward_iter = value
            for _p in self.parties:
                _p.is_first_forward_iter = value

        def train(self,*args,**kwargs):
            for p in self.parties:
                p.train(*args,**kwargs)

        def eval(self):
            for p in self.parties:
                p.eval()

        def get_base_model(self):
            model_folder = get_model_folder()
            if not self.args.model_path and self.args.model_path:
                raise ValueError('model_path must not be empty and should contain /')
            base_model = self.args.model_path[len(model_folder)+1:]
            return base_model

        def create_model_id(self):
            base_model = self.get_base_model()
            job_id = self.job_id
            if job_id is None:
                current_datetime = datetime.now()
                job_id = current_datetime.strftime("%Y%m%d%H:%M:%S")
            return f'{base_model}_{job_id}'

        def train_vfl(self, model_id=None, save_model=False, **kwargs):  # def train(self):
            training_args = vfl_basic_config.vfl_training_config.training_args
            if self.args.model_type.lower() == 'qwen2':
                # 创建 TensorBoard 摘要写入器
                tensorboard_writer = SummaryWriter(training_args.logging_dir)  # training_args.logging_dir)

            print_every = 1

            for ik in range(self.k):
                self.parties[ik].prepare_data_loader()

            test_acc = 0.0
            # Early Stop
            early_stop_threshold = self.args.early_stop_threshold
            last_loss = 1000000
            early_stop_count = 0
            LR_passive_list = []
            LR_active_list = []

            self.num_total_comms = 0
            total_time = 0.0
            flag = 0
            self.current_epoch = 0

            start_time = time.time()
            optimize_step = 0

            data_record = pd.DataFrame(columns=['Epoch', 'train_loss', 'train_acc', 'test_acc'])
            # if (not is_test) and (self.args.model_type.lower() == 'qwen2'):
            #     self.eval()
            #     with torch.no_grad():
            #         _exp_result, test_acc = self.inference()
            #     tensorboard_writer.add_scalar('train/eval_loss', self._loss, 0)
            
            for i_epoch in range(self.epochs):
                self.train()
                self.current_epoch = i_epoch
                # if self.args.model_type.lower() == 'qwen2':
                #     tensorboard_writer.add_scalar('train/epoch', i_epoch, optimize_step)

                postfix = {'train_loss': 0.0, 'train_acc': 0.0, 'test_acc': 0.0}
                i = -1
                print_every = 1
                total_time = 0

                data_loader_list = [self.parties[ik].train_loader for ik in range(self.k - 1)]
                for parties_data in tqdm(zip(*data_loader_list), desc=f'Epoch {i_epoch}/{self.epochs - 1}'):
                    ############ Allocate Data #################
                    _parties_data = []
                    for party_id in range(len(parties_data)):  # parties_data[party_id]: list of bs
                        batch_input_dicts = []
                        batch_label = []
                        for bs_id in range(len(parties_data[party_id])):
                            batch_input_dicts.append(parties_data[party_id][bs_id][0]) # Input Dict
                            if type(parties_data[party_id][bs_id][1]) != str: # Label
                                batch_label.append(parties_data[party_id][bs_id][1].tolist())
                            else:
                                batch_label.append(parties_data[party_id][bs_id][1])
                        _parties_data.append([batch_input_dicts, batch_label])
                    self.parties_data = _parties_data
                    
                    self.gt_one_hot_label = [] # party_number * party_label
                    for party_id in range(len(parties_data)):
                        if self.args.model_architect == 'CLS' and self.args.num_classes > 1:  # classification
                            self.gt_one_hot_label.append(self.label_to_one_hot(self.parties_data[party_id][1], self.args.num_classes))
                        elif self.args.model_architect == 'TQA':
                            self.gt_one_hot_label.append(list(self.parties_data[party_id][1]))
                        else:
                            self.gt_one_hot_label.append(self.parties_data[party_id][1])
                    
                    i += 1

                    # passive party call active party global model to a training mode
                    self._communication.send_global_model_train_message()

                    # ====== train batch (start) ======
                    if self.args.need_first_epoch_state and i == 0 and i_epoch == 0:
                        self.first_epoch_state = self.save_state(True)

                    enter_time = time.time()
                    self.loss, self.train_acc = self.train_batch(self.parties_data, self.gt_one_hot_label)
                    exit_time = time.time()

                    if self.args.need_first_epoch_state and i == 0 and i_epoch == 0:
                        self.first_epoch_state.update(self.save_state(False))
                    
                    
                    # ====== train batch (end) ======
                    
                    total_time += (exit_time - enter_time)
                    optimize_step += 1
                    self.num_total_comms = self.num_total_comms + 1
                    # if self.num_total_comms % 10 == 0:
                    #     print(f"total time for {self.num_total_comms} communication is {total_time}")
                    self.current_step = self.current_step + 1

                    del (parties_data)

                    if self.epochs == 1 :
                        break

                # LR decay
                self.LR_Decay(i_epoch)

                # if self.args.apply_adversarial:
                #     print(f'global_loss={self.parties[0].global_loss} adversarial_model_loss:{self.parties[0].adversarial_model_loss.item()} adversary_attack_loss:{self.parties[0].adversary_attack_loss.item()}')
                # if self.args.apply_mid:
                #     print(f'global_loss={self.parties[0].global_loss},head_mid_loss={self.parties[0].head_mid_loss}')

                self.final_epoch = i_epoch + 1

                # Early Stop
                if self.loss >= last_loss:
                    early_stop_count = early_stop_count + 1
                if early_stop_count >= early_stop_threshold:
                    self.final_epoch = i_epoch + 1
                    break
                last_loss = min(last_loss, self.loss)

                if self.args.task_type == 'SequenceClassification' and self.args.num_classes == 1:
                    exp_result = 'Epoch {}% \t train_loss:{:.2f} train_mse:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc[0])
                else:
                    exp_result = 'Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc)
                print(exp_result)
            
            ################### Evaluation on Test Data ###################
            print("----- Evaluation ------")
            self.test_acc = 0
            # if (i + 1) % print_every == 0:
            self.parties[self.k - 1].eval()
            self.eval()
            with torch.no_grad():
                _exp_result, self.test_acc = self.inference(need_save_state = self.args.need_final_epoch_state)

                postfix['train_loss'] = self.loss
                # postfix['train_acc'] = '{:.2f}%'.format(self.train_acc * 100)
                # postfix['test_acc'] = '{:.2f}%'.format(self.test_acc * 100)
                # postfix['test_auc'] = '{:.2f}%'.format(self.test_auc * 100)
                # postfix['test_mcc'] = '{:.2f}%'.format(self.test_mcc * 100)

                if self.args.task_type == 'SequenceClassification' and self.args.num_classes == 1:
                    exp_result = 'Epoch {}% \t train_loss:{:.2f} train_mse:{:.2f} test_mse:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc[0], self.test_acc[0])
                else:
                    exp_result = 'Epoch {}% \t train_loss:{:.2f} train_acc:{:.2f} test_acc:{:.2f}'.format(
                        i_epoch, self.loss, self.train_acc, self.test_acc)
                print(exp_result)


            # if self.args.need_final_epoch_state:
            # self.final_state=self.save_state(False)
            self.final_state.update(self.save_party_data())

            # if self.args.model_type.lower() == 'qwen2':
            #     tensorboard_writer.add_scalar('train/eval_loss', self._loss, optimize_step)

            # data_record.loc[len(data_record)] = [i_epoch, self.loss, self.train_acc, self.test_acc]

            
            
            try:
                self.save_pretrained(model_index=[1,2], model_id=model_id)
            except Exception as e:
                logger.warning(repr(e))
            
            self.training_time = total_time
            
            if self.args.task_type == 'SequenceClassification' and self.args.num_classes == 1:
                exp_result = f'|train_party_time={self.train_party_time}|training_time={total_time}|train_loss:{self.loss}|\
            train_mse={self.train_acc[0]}|train_pearson_corr={self.train_acc[1]}|train_spearmanr_corr={self.train_acc[2]}|\
            test_mse={self.test_acc[0]}|train_pearson_corr={self.test_acc[1]}|test_spearmanr_corr={self.test_acc[2]}|\
            final_epoch={self.final_epoch}'
            else:
                exp_result = f'|train_party_time={self.train_party_time}|training_time={total_time}|train_loss={self.loss}|train_acc={self.train_acc}|\
            test_acc={self.test_acc}|final_epoch={self.final_epoch}'

            
            result_path = f'exp_result/{self.args.dataset}/Q{str(self.args.Q)}/'
            model_name = self.args.model_list["name"].replace('/', '-')
            if self.args.pipeline == 'pretrained':
                filename = f'{self.args.defense_name}_{self.args.defense_param},pretrained_model={model_name}'
            else:
                filename = f'{self.args.defense_name}_{self.args.defense_param},finetuned_model={model_name}'
            result_file_name = result_path + filename + f'.csv'
            # print('Save csv to:', result_file_name)
            # data_record.to_csv(result_file_name)

            if self.args.apply_defense and save_model:
                if self.args.apply_mid or self.args.apply_adversarial:
                    self.save_defense_models()
            return exp_result, self.test_acc, total_time  # , self.stopping_iter, self.stopping_time, self.stopping_commu_cost

        def dict_deepcopy(self, origin_dict, device = None):
            new_dict = {}

            for _key in origin_dict.keys():
                # print(f'{_key}:{type(origin_dict[_key])}')
                if origin_dict[_key]!=None:
                    if device == 'None':
                        new_dict[_key] = copy.deepcopy(origin_dict[_key].detach())
                    else:
                        new_dict[_key] = copy.deepcopy(origin_dict[_key].detach()).to(device)
                else:
                    new_dict[_key] = None

            return new_dict
            
        def save_state(self, BEFORE_MODEL_UPDATE=True):
            if self.args.need_save_state:
                if BEFORE_MODEL_UPDATE:
                    # print('save:',self.parties[0].local_model_tail.head_layer.weight[0,:5])
                    return {
                        "local_model_head": copy.deepcopy(self.parties[0].local_model).to("cpu") if self.parties[0].local_model != None else None,
                        "local_model_tail": copy.deepcopy(self.parties[0].local_model_tail).to("cpu") if self.parties[0].local_model_tail != None else None,
                        # "vis_processors": copy.deepcopy(self.parties[0].vis_processors) if self.parties[0].vis_processors != {} else None,
                        
                        "active_model_body": copy.deepcopy(self.parties[self.args.k - 1].global_model).to("cpu") if self.parties[self.args.k - 1].global_model != None else None,

                        "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in
                                        range(self.args.k)] + [
                                        str(type(self.parties[self.args.k - 1].global_model)).split('.')[-1].split('\'')[
                                            -2]]

                    }
                else:
                    # record info exhcange between attacker(active party) and victim(passive party 0 by default)
                    return {
                        # Batch Label
                        "label": copy.deepcopy(self.gt_one_hot_label),
                        # Batch Data
                        "batch_data": copy.deepcopy(self.parties_data),

                        # Transmission
                        "passive_predict": self.dict_deepcopy(self.parties[0].output_tensors),
                        "passive_predict_attention_mask": self.dict_deepcopy(self.parties[0].output_attention_mask) ,

                        "active_predict": self.dict_deepcopy(self.parties[self.args.k - 1].output_tensors[0]) ,
                        "active_predict_attention_mask": self.dict_deepcopy(self.parties[self.args.k - 1].output_attention_mask[0]) ,
                        
                        "local_gradient": copy.deepcopy(self.parties[0].local_gradient.detach()) if self.parties[0].local_gradient!= None else None,
                        "global_gradient": copy.deepcopy(self.parties[self.args.k - 1].global_gradient_dict[0].detach()) if self.parties[self.args.k - 1].global_gradient_dict[0] != None else None,
                        
                        # Gradient
                        "local_model_head_gradient": copy.deepcopy(self.parties[0].weights_grad_a),
                        "local_model_tail_gradient": copy.deepcopy(self.parties[0].weights_grad_a_tail) ,
                        "global_model_body_gradient": copy.deepcopy(self.parties[self.args.k - 1].weights_grad_a_list[0]) ,
                        
                        # Result
                        "real_generation_result": self.real_generation_result,
                        "train_acc": copy.deepcopy(self.train_acc),
                        "loss": copy.deepcopy(self.loss),
                        
                        # "global_pred": self.parties[self.k - 1].global_output,
                        # "final_model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                        # "final_global_model": copy.deepcopy(self.parties[self.args.k - 1].global_model),
                    }
        
        def save_element(self, element_name):
            if element_name == "label": 
                return {element_name: copy.deepcopy(self.gt_one_hot_label)}
            
            elif element_name == "batch_data": 
                return {element_name: copy.deepcopy(self.parties_data)}
            
            elif element_name == "first_iter_passive_predict": 
                return {element_name: self.dict_deepcopy(self.parties[0].output_tensors)}
            
            elif element_name == "first_iter_passive_predict_attention_mask": 
                return {element_name: self.dict_deepcopy(self.parties[0].output_attention_mask)}
            
            elif element_name == "active_predict_list": 
                return {element_name: [self.dict_deepcopy(self.parties[1].output_tensors)] }
            
            elif element_name == "active_predict_attention_mask_list": 
                return {element_name: [self.dict_deepcopy(self.parties[1].output_attention_mask)] }
            
            elif element_name == "real_generation_result": 
                return {element_name: self.real_generation_result}
            
            else:
                assert 1>2, f"{element_name} has no save method"

        def save_party_data(self):
            return {
                # "aux_data": [copy.deepcopy(self.parties[ik].aux_data) for ik in range(self.k)],
                "train_data": [copy.deepcopy(self.parties[ik].train_data) for ik in range(self.k)],
                "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k - 1)],

                # "aux_dst": [self.parties[ik].aux_dst for ik in range(self.k)],
                # "train_dst": [self.parties[ik].train_dst for ik in range(self.k)],
                # "test_dst": [self.parties[ik].test_dst for ik in range(self.k)],

                # "aux_label": [copy.deepcopy(self.parties[ik].aux_label) for ik in range(self.k)],
                "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
                "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k - 1)],

                # "aux_attribute": [copy.deepcopy(self.parties[ik].aux_attribute) for ik in range(self.k)],
                # "train_attribute": [copy.deepcopy(self.parties[ik].train_attribute) for ik in range(self.k)],
                # "test_attribute": [copy.deepcopy(self.parties[ik].test_attribute) for ik in range(self.k)],

                # "aux_loader": [self.parties[ik].aux_loader for ik in range(self.k)],
                # "train_loader": [self.parties[ik].train_loader for ik in range(self.k)],
                # "test_loader": [self.parties[ik].test_loader for ik in range(self.k)],

                "batchsize": self.args.batch_size,
                "num_classes": self.args.num_classes
            }

        def save_defense_models(self):
            dir_path = self.exp_res_dir + f'/defense_models/'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # if self.args.apply_mid:
            #     file_path = dir_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'
            #     with open(file_path, 'wb') as f:
            #         pickle.dump(self.parties[0].mid_model, f)

            #     file_path = dir_path + f'head_{self.args.defense_name}_{self.args.defense_configs}.pkl'
            #     with open(file_path, 'wb') as f:
            #             pickle.dump(self.parties[1].global_model.head_layer, f)

            if self.args.apply_adversarial:
                file_path = dir_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'
                with open(file_path, 'wb') as f:
                    pickle.dump(self.parties[0].adversarial_model, f)

                file_path = dir_path + f'head_{self.args.defense_name}_{self.args.defense_configs}.pkl'
                with open(file_path, 'wb') as f:
                        pickle.dump(self.parties[1].global_model.head_layer, f)

            # with open('my_model.pkl', 'rb') as f:
            #     model = pickle.load(f)
            # torch.save(self.parties[0].mid_model.state_dict(),self.trained_models["model_names"],
            #         file_path)

        def save_trained_models(self):
            dir_path = self.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.apply_trainable_layer}_epoch{self.epochs}/'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            if self.args.apply_defense:
                file_path = dir_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'
            else:
                file_path = dir_path + 'NoDefense.pkl'
            torch.save(
                ([self.trained_models["model"][i].state_dict() for i in range(len(self.trained_models["model"]))],
                 self.trained_models["model_names"]),
                file_path)

        def save_pretrained(self,model_index,model_id,**kwargs):
            """
            :param model_index:
            :param kwargs: refer PretrainedModel
            :return:
            """
            if model_id is None:
                model_id = self.create_model_id()
            logger.info(f"save trained models {model_id}")
            for p in self.parties:
                p.save_pretrained(model_index,
                                  model_id=model_id,
                                  model_folder=get_model_folder(),
                                  **kwargs)

        def evaluate_attack(self):
            self.attacker = AttackerLoader(self, self.args)
            if self.attacker != None:
                attack_acc = self.attacker.attack()
            return attack_acc

        def launch_defense(self, gradients_list, _type):

            if _type == 'gradients':
                return apply_defense(self.args, _type, gradients_list)
            elif _type == 'pred':
                return apply_defense(self.args, _type, gradients_list)
            else:
                # further extention
                return gradients_list

        def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
            try:
                if self.args.vfl_model_slice_num == 2:
                    if self.args.finetune_name == 'LoRA':
                        return self.parties[-1].global_model.model._validate_model_kwargs(model_kwargs)
                    else:
                        return self.parties[-1].global_model._validate_model_kwargs(model_kwargs)
                else:
                    if self.args.finetune_name == 'LoRA':
                        return self.parties[0].local_model_tail.model._validate_model_kwargs(model_kwargs)
                    else:
                        return self.parties[0].local_model_tail._validate_model_kwargs(model_kwargs)
            except:
                pass

        def _validate_model_class(self):
            pass

        # def prepare_inputs_for_generation(self, *args, **model_kwargs):
        #     return super().prepare_inputs_for_generation(*args, **model_kwargs)

        # def prepare_inputs_for_generation(self, input_ids, **model_kwargs):
        #     if vfl_basic_config.num_of_slice == 3:
        #         return super().prepare_inputs_for_generation(input_ids, **model_kwargs)
        #     else:
        #         # return self.parties[-1].global_model.prepare_inputs_for_generation(input_ids=input_ids, **model_kwargs)
        #         return self.parties[0].local_model.prepare_inputs_for_generation(input_ids=input_ids, **model_kwargs)

        def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs,
                                                           model_input_name: Optional[str] = None):
            if self.args.vfl_model_slice_num == 3:
                return super()._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor=inputs_tensor,
                                                                              model_kwargs=model_kwargs,
                                                                              model_input_name=model_input_name)
            else:
                return self.parties[-1].global_model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor=inputs_tensor, model_kwargs=model_kwargs, model_input_name=model_input_name)

        def __call__(self, **kwargs):
            return self.forward(**kwargs)

        def _clear_past_key_values(self):
            for i in range(self.k - 1):
                self.parties[i].local_model._clear_past_key_values()
            # todo: clear global model's past_key_values
            # self.parties[-1].global_model._clear_past_key_values()

        @property
        def passive_party(self):
            return self.parties[0]

        def gen(self, prompt: str = '你是谁',
                prompt_system: str = "现在你要扮演皇帝身边的女人--甄嬛"):
            start_time = time.time()
            tokenizer = self.args.tokenizer
            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt}
            ]

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            model_inputs = tokenizer([text], return_tensors="pt").to(self.device)
            generated_ids = self.generate(
                model_inputs.input_ids,
                max_new_tokens=20,
                use_cache=False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            end_time = time.time()
            logger.info(f"Took: {end_time - start_time} s\nGenerated: {response}")
            return response

        def shift_logits_loss(self, logits, labels, model_config):
            from torch.nn import CrossEntropyLoss

            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.args.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            return loss

    return MainTaskVFL_LLM
