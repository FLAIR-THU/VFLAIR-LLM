import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
import itertools

from evaluates.attacks.attacker import Attacker
from utils.basic_functions import append_exp_res # cross_entropy_for_onehot
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
torch.backends.cudnn.enabled = False
from peft import LoraConfig, get_peft_model

# from nltk.translate.bleu_score import sentence_bleu
# from nltk.translate.bleu_score import SmoothingFunction
from sacrebleu import sentence_bleu 
from nltk.translate.meteor_score import meteor_score

from dataset.party_dataset import *
from load.LoadModels import load_llm_slice

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * pred, 1))

class ResultReconstruction(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        self.top_vfl = top_vfl
        self.vfl_info = top_vfl.final_state
        
        # attack configs
        print(args.attack_configs.keys())
        self.party = args.attack_configs['party'] # parties that launch attacks , default 1(active party attack)
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.attack_batch_size = args.attack_configs['batch_size']
        self.attack_sample_num = args.attack_configs['attack_sample_num']
        self.aux_data_percentage = args.attack_configs['aux_data_percentage'] if 'aux_data_percentage' in args.attack_configs else None
        self.aux_data_num = args.attack_configs['aux_data_num'] if 'aux_data_num' in args.attack_configs else None
        self.origin_model_path = args.attack_configs['origin_model_path'] if 'origin_model_path' in args.attack_configs else None
        
        self.use_lora = args.attack_configs['use_lora'] if 'use_lora' in args.attack_configs else False
        self.lora_configs = args.attack_configs['lora_configs'] if 'lora_configs' in args.attack_configs else {}
        
        self.criterion = nn.CrossEntropyLoss()
        print('Check:',args.model_folder+'/test_sample_state_list.pth')
        if os.path.exists(args.model_folder+'/test_sample_state_list.pth'):
            print('Load test_sample_state_list from:',args.model_folder+'/test_sample_state_list.pth')
            self.test_sample_state_list = torch.load(args.model_folder+'/test_sample_state_list.pth', map_location='cpu')
        else:
            print('Load test_sample_state_list from vfl')
            if len(top_vfl.test_sample_state_list) >= self.attack_sample_num:
                self.test_sample_state_list = top_vfl.test_sample_state_list[:self.attack_sample_num]
            else:
                self.test_sample_state_list = top_vfl.test_sample_state_list
        print('self.test_sample_state_list:',len(self.test_sample_state_list))
        assert len(self.test_sample_state_list) > 0
        
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k
        self.file_name = 'attack_result.txt'
       
    def set_seed(self, seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def cal_loss(self, pred, label, test=False):
        # print(f'=== cal_loss label:{label.shape} pred:{type(pred)}{pred.logits.shape} ===')
        # ########### Normal Loss ###############
        if self.args.model_architect == 'CLS':  
            pooled_logits = pred.logits # [bs, num_labels]
            labels = torch.argmax(label,-1) # [bs, num_labels] -> [bs]
            
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

        elif self.args.model_architect == 'CLM':  
            lm_logits = pred.logits  
            labels = torch.tensor(label).to(lm_logits.device)
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

            return loss
        
    def attack(self):
        if 'attack_seed' in self.args.attack_configs.keys():
            attack_seed = self.args.attack_configs['attack_seed']
        else:
            attack_seed = self.args.current_seed
        self.set_seed(attack_seed)
        


        for attacker_ik in self.party: # attacker party #attacker_ik
            assert attacker_ik == (self.k - 1), 'Only Active party launch input inference attack'
            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)
            index = attacker_ik

            ####### collect necessary information #######
            completed_model_head_path = self.args.model_folder+'/completed_model_head/'
            completed_model_tail_path = self.args.model_folder+'/completed_model_tail/'
            
            # if os.path.exists(completed_model_head_path) and os.path.exists(completed_model_tail_path):
            #     dummy_model_tail = load_llm_slice(args=self.args,slice_index = 2,model_path=completed_model_head_path).to(self.args.device)
            #     dummy_model_head = load_llm_slice(args=self.args,slice_index = 0,model_path=completed_model_tail_path).to(self.args.device)
            # else:
            if self.origin_model_path:
                dummy_model_tail = load_llm_slice(args=self.args,slice_index = 2,model_path=self.origin_model_path).to(self.args.device)
                dummy_model_head = load_llm_slice(args=self.args,slice_index = 0,model_path=self.origin_model_path).to(self.args.device)
            else:
                dummy_model_tail = load_llm_slice(args=self.args,slice_index = 2).to(self.args.device)
                dummy_model_head = load_llm_slice(args=self.args,slice_index = 0).to(self.args.device)
            
            if self.use_lora:
                lora_config = LoraConfig(
                    **self.lora_configs
                )
                dummy_model_head = get_peft_model(dummy_model_head, lora_config)
                dummy_model_tail = get_peft_model(dummy_model_tail, lora_config)
                print('Apply LoRA on dummy_model_head/dummy_model_tail')
                dummy_model_tail.print_trainable_parameters()
                dummy_model_head.print_trainable_parameters()
                
            optimizer = torch.optim.Adam([
                {'params': dummy_model_tail.parameters()},
                {'params': dummy_model_head.parameters()}], lr=self.lr)
            # ema_optimizer = WeightEMA(model, ema_model, lr=self.lr, alpha=self.ema_decay)  # ema_decay = 0.999

            enter_time = time.time()
            
            ############ Begin Attack ###############
            # data preparation
            if self.aux_data_num == None:
                assert self.aux_data_percentage != None, "must specify aux_data_num or aux_data_percentage"
                self.aux_data_num = int(len(self.vfl_info["train_data"][0]) * self.aux_data_percentage)
            
            indices = np.arange(len(self.vfl_info["train_data"][0]) )
            aux_indices = list( np.random.choice(indices, size=self.aux_data_num, replace=False).astype(int) )
            aux_data = [self.vfl_info["train_data"][0][i] for i in aux_indices]
            aux_label = [self.vfl_info["train_label"][0][i] for i in aux_indices]
           
            
            if self.args.dataset == 'Lambada' or self.args.dataset == 'Lambada_test':
                aux_dataset = LambadaDataset_LLM(self.args, aux_data, aux_label, 'train')
            elif self.args.dataset == 'TextVQA' or self.args.dataset == 'TextVQA-test':
                aux_dataset = TextVQADataset_train(self.args, aux_data, aux_label, vis_processor,'train')
            elif self.args.dataset == 'GMS8K' or self.args.dataset == 'GMS8K-test':
                aux_dataset = GSMDataset_LLM(self.args, aux_data, aux_label, 'train')
            elif self.args.dataset in ['CodeAlpaca','Alpaca','Alpaca-test']:
                aux_dataset = AlpacaDataset_LLM(self.args, aux_data, aux_label, 'train')
            else:
                aux_dataset = PassiveDataset_LLM(self.args, aux_data, aux_label)
            aux_data_loader = DataLoader(aux_dataset, batch_size= self.attack_batch_size ,collate_fn=lambda x:x )
            torch.cuda.empty_cache()
            
            def move_dict_to_device(d, device):
                for key in d:
                    if isinstance(d[key], torch.Tensor):
                        d[key] = d[key].to(device)
                return d
            
            print(f'=== Phase1: Begin Model Completion Training on {len(aux_data)} aux data samples ===')
            early_stop = 0
            best_epoch_loss = 100000
            epoch_loss = 100000
            final_epoch = 0
            for epoch in range(self.epochs):  
                total_iter = len(aux_data_loader)
                epoch_loss = 0
                for origin_input in aux_data_loader:
                    batch_input_dicts = []
                    batch_label = []
                    for bs_id in range(len(origin_input)):
                        batch_input_dicts.append(origin_input[bs_id][0])
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


                    self.top_vfl.set_is_first_forward_epoch(1)
                    
                    # # real received intermediate result
                    # self.top_vfl.parties[0].obtain_local_data(data_inputs)
                    # self.top_vfl.parties[0].gt_one_hot_label = batch_label
                    # # passive party do local pred
                    # passive_party_pred = self.top_vfl.pred_transmit(use_cache=False)
                    data_inputs = move_dict_to_device(data_inputs, dummy_model_head.device)
                    passive_party_pred = dummy_model_head(**data_inputs)
                    
                    # passive party inform active party to do global pred
                    active_party_pred = self.top_vfl.global_pred_transmit(passive_party_pred, use_cache=False)

                    # dummy final prediction
                    active_party_pred = move_dict_to_device(active_party_pred, dummy_model_tail.device)
                    dummy_final_pred = dummy_model_tail(**active_party_pred)
                    
                    self.top_vfl._clear_past_key_values()
                    
                    # batch_label [bs * token_id_list]
                    # dummy_final_pred.logits torch.size(bs, seq_len)
                    loss = self.cal_loss(dummy_final_pred, torch.tensor(batch_label))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                
                print('\nEpoch: [%d | %d] Loss: %f' % (epoch + 1, self.epochs, epoch_loss/total_iter))
                if epoch_loss > best_epoch_loss:
                    early_stop = early_stop + 1
                    if early_stop >=2:
                        break
                else:
                    early_stop = 0
                best_epoch_loss = min(epoch_loss, best_epoch_loss)
                final_epoch = epoch
            print(f'Training Complete')
            
            # print(f'Save completed model into:',)
            # if not os.path.exists(completed_model_head_path):
            #     os.makedirs(completed_model_head_path)
            # if not os.path.exists(completed_model_tail_path):
            #     os.makedirs(completed_model_tail_path)
            # dummy_model_head.save_pretrained(completed_model_head_path)
            # dummy_model_tail.save_pretrained(completed_model_tail_path)
            
            ## begin reconstruction
            sample_num = len(self.test_sample_state_list)
            print(f'=== Phase2: Test Result Reconstruction on {sample_num} test batch samples ===')
            
            bleu_gen_mean = 0
            bleu_label_mean = 0
            meteor_gen_mean = 0
            meteor_label_mean = 0
            total_sample_count = 0
            
            id = 0
            for sample_info in self.test_sample_state_list:
                id = id+1
            
                label = sample_info['label'][0][0] # list: party_number * [party_label tensor]
                label_text = self.args.tokenizer.decode(label.squeeze(),skip_special_tokens=True)
                
                real_generate_result = sample_info['real_generation_result'] # torch.tensor (1, seq_len)
                real_generated_text = self.args.tokenizer.decode(real_generate_result.squeeze(),skip_special_tokens=True)
                
                # attainable intermediate results 
                ### prediction result of model_1(body)
                # [1, len, h] [1, len+1 h]...
                tail_input_embed_list = [ active_predict[1] for active_predict in sample_info['active_predict_list']]
                # [1, 1, 1, len] [1, 1, 1, len+1]...
                tail_input_attn_mask_list = [ active_predict_attention_mask[1] \
                    if active_predict_attention_mask[1] is not None else None \
                        for active_predict_attention_mask in sample_info['active_predict_attention_mask_list'] ]
                seq_len = len(tail_input_embed_list)
                
                # print('tail_input_embed_list:',len(tail_input_embed_list))
                # print(tail_input_embed_list[0].shape, tail_input_embed_list[1].shape)
                
                # ### input of model_1(body)
                # # [1, len, h] [1, len+1 h]...
                # body_input_embed_list = [ active_input[1] for active_input in sample_info['active_input_list']]
                # # [1, 1, 1, len] [1, 1, 1, len+1]...
                # body_input_attn_mask_list = [ active_input_attention_mask[1] \
                #     if active_input_attention_mask[1] is not None else None \
                #         for active_input_attention_mask in sample_info['active_input_attention_mask_list'] ]
                
                # dummy_generated_token_ids = [] 
                # for (tail_input_embed,tail_input_attn_mask) in zip(tail_input_embed_list,tail_input_attn_mask_list):
                #     prev_len = tail_input_embed.shape[1] # [1, 65, 768]  65
                    
                #     tail_input_attn_mask = tail_input_attn_mask.to(dummy_model_tail.device) if tail_input_attn_mask!= None else None
                #     dummy_output_logits = dummy_model_tail(inputs_embeds = tail_input_embed.to(dummy_model_tail.device), \
                #             attention_mask = tail_input_attn_mask)['logits'] # 1, seq_len, vocab_dim
                #     del tail_input_embed
                #     del tail_input_attn_mask
                    
                #     current_text_ids = torch.argmax(dummy_output_logits, dim=-1)[0]
                #     print('approx origin input--current_text_ids:',current_text_ids.shape)
                #     current_text = self.args.tokenizer.decode(current_text_ids)
                #     print(current_text)
                #     # dummy_next_token_id = torch.argmax(dummy_output_logits, dim=-1)[0,-1]
                #     # input_ids = torch.full(
                #     #     (1, prev_len + 1),
                #     #     fill_value=self.args.tokenizer.pad_token_id,  # 或用其他占位符
                #     #     device=dummy_model_head.device
                #     # )
                #     # input_ids[:, -1] = dummy_next_token_id  # 最后一位设为 dummy token
                #     # dummy_head_intermediate = dummy_model_head(input_ids=input_ids)['inputs_embeds']

                #     assert 1>2

                # _dummy_generation_result = self.args.tokenizer.decode(torch.tensor(dummy_generated_token_ids),skip_special_tokens=True)
                # print(torch.tensor(dummy_generated_token_ids).shape)
                # print('_dummy_generation_result:\n',dummy_generated_token_ids)
                
                last_tail_input_embed = tail_input_embed_list[-1].to(dummy_model_tail.device)
                last_tail_input_attn_mask = tail_input_attn_mask_list[-1].to(dummy_model_tail.device) if tail_input_attn_mask_list[-1]!= None else None
                dummy_output_logits = dummy_model_tail(inputs_embeds = last_tail_input_embed.to(dummy_model_tail.device), \
                        attention_mask = last_tail_input_attn_mask)['logits'] # 1, seq_len, vocab_dim
                
                dummy_generated_token_ids = torch.argmax(dummy_output_logits, dim=-1).squeeze()[-seq_len:]
                
                dummy_generation_text = self.args.tokenizer.decode(dummy_generated_token_ids,skip_special_tokens=True)
                label_text = self.args.tokenizer.decode(label,skip_special_tokens=True)
                real_generation_text = self.args.tokenizer.decode(real_generate_result.squeeze(),skip_special_tokens=True)
                # print('dummy_generation_text:\n',dummy_generation_text)
                # print('label_text:\n',label_text)
                # print('real_generation_text:\n',real_generation_text)

                label_tokens = label_text.split() #self.args.tokenizer.convert_ids_to_tokens(label.squeeze(), skip_special_tokens=True)  # 目标句子的 token 列表
                real_generated_tokens = real_generation_text.split() #self.args.tokenizer.convert_ids_to_tokens(real_generate_result.squeeze(), skip_special_tokens=True)  # 目标句子的 token 列表
                dummy_generated_tokens = dummy_generation_text.split() #self.args.tokenizer.convert_ids_to_tokens(dummy_generated_token_ids, skip_special_tokens=True)  # 生成句子的 token 列表
                # print('real_generated_tokens:',real_generated_tokens)
                # print('dummy_generated_tokens:',dummy_generated_tokens)
                # print('label_tokens:',label_tokens)
                
                # old_bleu_gen = sentence_bleu([real_generated_tokens], dummy_generated_tokens)
                # old_bleu_label = sentence_bleu([label_tokens], dummy_generated_tokens)
                # print('origin bleu gen_score:',gen_score,' label_score:',label_score)
                
                bleu_gen = 0.01 * sentence_bleu(dummy_generation_text, [real_generation_text]).score
                bleu_label = 0.01 * sentence_bleu(dummy_generation_text, [label_text]).score
                meteor_gen = meteor_score([real_generated_tokens], dummy_generated_tokens)
                meteor_label = meteor_score([label_tokens], dummy_generated_tokens)
                
                # print('new bleu gen_score:',gen_score,' label_score:',label_score)
                # print('meteor_gen:',meteor_gen,' meteor_label:',meteor_label)
                
                bleu_gen_mean += bleu_gen
                bleu_label_mean += bleu_label
                meteor_gen_mean += meteor_gen
                meteor_label_mean += meteor_label
                total_sample_count += 1
            
            bleu_gen_mean = bleu_gen_mean/total_sample_count
            bleu_label_mean = bleu_label_mean/total_sample_count
            meteor_gen_mean = meteor_gen_mean/total_sample_count
            meteor_label_mean = meteor_label_mean/total_sample_count

            exit_time = time.time()

            attack_total_time = exit_time - enter_time

            print('bleu_gen_mean:',bleu_gen_mean,'  bleu_label_mean:',bleu_label_mean)
            print('meteor_gen_mean:',meteor_gen_mean,'  meteor_label_mean:',meteor_label_mean)

        attack_info = f'attack_seed:{attack_seed} aux_data_num:{self.aux_data_num} aux_data_percentage:{self.aux_data_percentage} use_lora:{self.use_lora} attack_lr:{self.lr} attack_epoch:{self.epochs} attack_final_epoch:{final_epoch} attack_bs:{self.attack_batch_size} attack_loss:{epoch_loss}'
        print(attack_info)
        append_exp_res(self.args.exp_res_path, attack_info)
        
        attack_result = {
            "bleu_gen":bleu_gen_mean, "bleu_label":bleu_label_mean,
            "meteor_gen":meteor_gen_mean, "meteor_label":meteor_label_mean,
        }
    
        return attack_result, attack_total_time
        # return recovery_history
