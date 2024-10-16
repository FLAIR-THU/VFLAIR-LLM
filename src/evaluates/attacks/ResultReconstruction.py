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
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']

        self.early_stop = args.attack_configs['early_stop'] if 'early_stop' in args.attack_configs else 0
        self.early_stop_threshold = args.attack_configs[
            'early_stop_threshold'] if 'early_stop_threshold' in args.attack_configs else 1e-7
        if self.args.model_architect != 'CLM':
            self.label_size = args.num_classes
        else:
            self.label_size = args.model_config.vocab_size

       
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
        self.set_seed(123)


        for attacker_ik in self.party: # attacker party #attacker_ik
            assert attacker_ik == (self.k - 1), 'Only Active party launch input inference attack'
            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)
            index = attacker_ik

            ####### collect necessary information #######
            dummy_model_head = load_llm_slice(args=self.args,slice_index = 0).to(self.args.device)
            dummy_model_tail = load_llm_slice(args=self.args,slice_index = 2).to(self.args.device)

            class Generator(type(dummy_model_tail)):
                def __init__(self, dummy_model_tail, global_model ,**kwargs):
                    super().__init__(dummy_model_tail.config, **kwargs)
                    self.global_model = global_model
                
                def forward(self,
                        **kwargs):
                    # if head_output == None:
                    head_output = dummy_model_head.forward(**kwargs)

                    body_output = self.global_model.forward(**head_output)
                    #
                    final_output = dummy_model_tail.forward(**body_output)
                    return final_output

            dummy_generator = Generator(dummy_model_tail= dummy_model_tail,\
                global_model = self.top_vfl.parties[1].global_model).to(dummy_model_tail.device)

            batch_data = self.vfl_info['batch_data']
            self.top_vfl.
            active_input_embed = self.vfl_info['passive_predict'][0].to(dummy_model_tail.device)
            active_input_attn_mask = self.vfl_info['passive_predict_attention_mask'][0].to(dummy_model_tail.device)
            
            output_logits = self.top_vfl.parties[0].local_model_tail(inputs_embeds = active_input_embed, \
                    attention_mask = active_input_attn_mask)['logits']
            print('0 output_logits:',output_logits.shape) # bs seq_len, vocab_dim
            print(output_logits[0,:2,:5])
            
            output_logits = dummy_model_tail(inputs_embeds = active_input_embed, \
                    attention_mask = active_input_attn_mask)['logits']
            print('output_logits:',output_logits.shape) # bs seq_len, vocab_dim
            print(output_logits[0,:2,:5])
            # input_ids = torch.argmax(output_logits, dim=-1)
            # print('input_ids:',input_ids.shape)

            real_output_logits = self.vfl_info['passive_predict'][2]
            print('real_output_logits:',real_output_logits.shape) # bs seq_len, vocab_dim
            print(real_output_logits[0,:2,:5])
            assert 1>2

            input_text = self.args.tokenizer.batch_decode(input_ids)
            print(input_text)

            dummy_generate_result = dummy_generator.generate(inputs = input_ids,\
                generation_config = self.top_vfl.generation_config)
            print('dummy_generate_result:',type(dummy_generate_result),dummy_generate_result.shape)
            dummy_generated_text = self.args.tokenizer.batch_decode(dummy_generate_result)
            print(dummy_generated_text)

            real_generate_result = self.vfl_info['real_generation_result']
            print('real_generate_result:',type(real_generate_result),real_generate_result.shape)
            real_generated_text = self.args.tokenizer.batch_decode(real_generate_result)
            print(real_generated_text)

            assert 1>2

            
            # target
            true_label = self.vfl_info['label'].to(self.device)  # CLM: bs, seq_len
            if self.args.model_architect == 'CLM': 
                true_label = true_label[:,-1]
            
            del self.vfl_info

            

            ################ Begin Attack ################
            start_time = time.time()
     
            print(f"sample_count = {true_label.size()[0]}, number of classes = {self.label_size}")
            sample_count = true_label.size()[0]
            
            rec_rate = 0

            # simulate model body forward 
            active_model_body_result = active_model_body(
                        inputs_embeds = passive_model_head_pred,
                        attention_mask= passive_model_head_attention_mask)
            if self.args.vfl_model_slice_num == 3:
                active_model_body_pred = active_model_body_result['inputs_embeds']
                active_model_body_attention_mask = active_model_body_result['attention_mask']

                passive_model_tail_result = passive_model_tail(
                        inputs_embeds = active_model_body_pred,
                        attention_mask= active_model_body_attention_mask)
                passive_model_tail_pred = passive_model_tail_result.logits
            else:
                active_model_body_pred = active_model_body_result['logits']
                print('active_model_body_pred:',active_model_body_pred.shape)
            
            # set fake label
            dummy_label =torch.zeros(sample_count, self.label_size).to(self.device)
            dummy_label.requires_grad = True
            print(f'dummy_label={dummy_label.shape} true_label={true_label.shape}')
            
            # # set optimizer
            optimizer = torch.optim.Adam([dummy_label],
                        lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  #
            # optimizer = torch.optim.Adam(
            #             itertools.chain([dummy_label], list(passive_model_tail.parameters())),
            #             lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  #
        
        

            min_loss = 100000
            early_stop = 0
            for iters in range(1, self.epochs + 1):
                optimizer.zero_grad()

                if self.args.vfl_model_slice_num == 3:
                    dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                    dummy_loss = self.cal_loss(passive_model_tail_pred, dummy_onehot_label)
                else:
                    dummy_loss = self.cal_loss(active_model_body_pred, dummy_label)

                dummy_dy_dx_a = torch.autograd.grad(dummy_loss, active_model_body_params, create_graph=True, allow_unused=True)
                # print(f'dummy_dy_dx_a:{type(dummy_dy_dx_a)} {len(dummy_dy_dx_a)}')

                # loss: L-L'
                grad_diff = 0
                for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                    if gx != None:
                        grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward(retain_graph=True)

                if iters == 1:
                    print('origin grad_diff:',grad_diff.item())

                if grad_diff.item() > min_loss:
                    early_stop += 1
                else:
                    early_stop = 0
                min_loss = min(grad_diff.item(), min_loss)

                if iters%50 == 0:
                    rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                    print('Iters',iters,' grad_diff:',grad_diff.item(),' rec_rate:',rec_rate)

                optimizer.step()

                if self.early_stop == 1:
                    if early_stop > self.early_stop_threshold:
                        break

            
            

            end_time = time.time()

            final_rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
            print('dummy_label:',torch.argmax(dummy_label, dim=-1) )
            print('true_label:',torch.argmax(true_label, dim=-1) )


            ########## Clean #########
            del (passive_model_head_pred)
            del (passive_model_head_attention_mask)
            del (original_dy_dx)
            del (active_model_body)
            del (dummy_label)

            

            print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf,time_used=%lf' % (
            sample_count, self.label_size, index, final_rec_rate, end_time - start_time))

            attack_total_time = end_time - start_time
            print(f"BLI, if self.args.apply_defense={self.args.apply_defense}")


        print("returning from BLI")
        return final_rec_rate, attack_total_time
        # return recovery_history
