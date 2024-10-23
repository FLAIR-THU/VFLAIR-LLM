import sys, os

sys.path.append(os.pardir)

import torch
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

from nltk.translate.bleu_score import sentence_bleu

# from models.llm_models.gpt2 import ModelPartitionPipelineGPT2
from load.LoadModels import load_llm_slice

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * pred, 1))

class ResultReconstruction(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.top_vfl = top_vfl
        self.vfl_info = top_vfl.final_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k
        self.party = args.attack_configs['party']  # parties that launch attacks

       
        self.criterion = cross_entropy_for_onehot
        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/BLR/'
        self.exp_res_path = ''

    def set_seed(self, seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        if len(gt_label.shape)==1:
            success = torch.sum(torch.argmax(dummy_label, dim=-1) == gt_label).item()
        else:
            success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()

        total = dummy_label.shape[0]
        return success / total

    def cal_loss(self, pred, label):
        # print(f'=== cal_loss label:{label.shape} {label.requires_grad} pred:{pred.shape} ===')
        if self.args.model_architect == 'CLM': # label: bs, vocab_size
            lm_logits = pred#.logits  # [bs, seq_len, vocab_size]
            next_token_logits = lm_logits[:, -1, :]
            # print(f'cal loss next_token_logits={next_token_logits.shape} label={real_label.shape}')
            loss = self.criterion(next_token_logits, label)
        elif self.args.model_architect == 'CLS': # label: bs, num_labels
            pooled_logits = pred#.logits # [bs, num_labels]
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(pooled_logits.view(-1, self.label_size), label.view(-1))
            loss = self.criterion(pooled_logits, label)

        
        return loss

    def attack(self):
        self.set_seed(self.args.current_seed)


        for attacker_ik in self.party: # attacker party #attacker_ik
            assert attacker_ik == (self.k - 1), 'Only Active party launch input inference attack'
            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)
            index = attacker_ik

            ####### collect necessary information #######
            dummy_model_tail = load_llm_slice(args=self.args,slice_index = 2).to(self.args.device)
            # dummy_model_head = load_llm_slice(args=self.args,slice_index = 0).to(self.args.device)
            
            batch_label = self.vfl_info['label']  # print(self.args.tokenizer.batch_decode(batch_label))
            batch_label_text = self.args.tokenizer.batch_decode(batch_label)

            real_generate_result = self.vfl_info['real_generation_result']
            real_generated_text = self.args.tokenizer.batch_decode(real_generate_result)

            # real_output_logits = self.vfl_info['passive_predict'][2]

            bs = len(real_generated_text)
            # attainable intermediate results
            # tail_input_embed = self.vfl_info['active_predict'][1].to(dummy_model_tail.device)
            # tail_input_attn_mask = self.vfl_info['active_predict_attention_mask'][1].to(dummy_model_tail.device)

            tail_input_embed_list = [ active_predict[1].to(dummy_model_tail.device) for active_predict in self.vfl_info['active_predict_list']]
            tail_input_attn_mask_list = [ active_predict_attention_mask[1].to(dummy_model_tail.device) for active_predict_attention_mask in self.vfl_info['active_predict_attention_mask_list']]
            
            del self.vfl_info

            enter_time = time.time()
            ############ Begin Attack ###############

            dummy_generated_token_ids = [[] for _i in range(bs)]
            for (tail_input_embed,tail_input_attn_mask) in zip(tail_input_embed_list,tail_input_attn_mask_list):

                dummy_output_logits = dummy_model_tail(inputs_embeds = tail_input_embed.to(dummy_model_tail.device), \
                        attention_mask = tail_input_attn_mask.to(dummy_model_tail.device))['logits'] # bs seq_len, vocab_dim
                del tail_input_embed
                del tail_input_attn_mask

                dummy_next_token_id = torch.argmax(dummy_output_logits, dim=-1)[:,-1]
                for _i in range(bs):
                    dummy_generated_token_ids[_i].append(dummy_next_token_id[_i].item())
            dummy_generated_tokens = self.args.tokenizer.batch_decode(dummy_generated_token_ids)
            
            exit_time = time.time()

            gen_mean_score = 0
            label_mean_score = 0
            for _i in range(bs):
                def wash(ids, target_token_id):
                    print('ids:',type(ids),len(ids))
                    print('target_token_id:',target_token_id)
                    while(target_token_id in ids):
                        ids.remove(target_token_id)
                    return ids
                
                def get_result(token_ids, washed_ids_list, tokenizer):
                    for washed_ids in washed_ids_list:
                        token_ids = wash(token_ids,washed_ids)
                    return tokenizer.convert_ids_to_tokens(token_ids)
                
                wash_list = [self.args.tokenizer.pad_token_id, self.args.tokenizer.eos_token_id, self.args.tokenizer.bos_token_id]
                dummy = get_result(dummy_generated_token_ids[_i], wash_list, self.args.tokenizer)
                real_gen = [ get_result(real_generate_result[_i].tolist(), wash_list, self.args.tokenizer) ]
                
                real_label = [ self.args.tokenizer.convert_ids_to_tokens(batch_label[_i].tolist()) ]
                
                # dummy = self.args.tokenizer.convert_ids_to_tokens(dummy_generated_token_ids[_i])
                # real_gen = [ self.args.tokenizer.convert_ids_to_tokens(real_generate_result[_i].tolist()) ]
                # real_label = [ self.args.tokenizer.convert_ids_to_tokens(batch_label[_i].tolist()) ]
                
                gen_score = sentence_bleu(real_gen, dummy)
                label_score = sentence_bleu(real_label, dummy)

                print('dummy:',dummy_generated_tokens[_i])
                print('real:',real_generated_text[_i])
                print('label:',batch_label_text[_i])


                print('gen_score:',gen_score,' label_score:',label_score)
                gen_mean_score += gen_score
                label_mean_score += label_score

            gen_mean_score = gen_mean_score/bs
            label_mean_score = label_mean_score/bs

            attack_total_time = exit_time - enter_time

            print('gen_mean_score:',gen_mean_score,'  label_mean_score:',label_mean_score)


        return {"gen_score":gen_mean_score, "label_score":label_mean_score}, attack_total_time
        # return recovery_history
