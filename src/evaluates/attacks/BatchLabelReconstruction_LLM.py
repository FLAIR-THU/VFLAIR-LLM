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

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * pred, 1))

def label_to_one_hot(target, num_classes=10):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        # print("use target itself", target.size())
        onehot_target = target.type(torch.float32)
    except:
        target = torch.unsqueeze(target, 1)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes).to(target.device)
        onehot_target.scatter_(1, target, 1)
    return onehot_target


class BatchLabelReconstruction_LLM(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.vfl_info = top_vfl.first_epoch_state
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

        self.dummy_active_top_trainable_model = None
        self.optimizer_trainable = None  # construct later
        self.dummy_active_top_non_trainable_model = None
        self.optimizer_non_trainable = None  # construct later
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
            # [Active Model Body] -> active_model_body_pred -> [Passive Model Tail] 
            # -> final_pred + true label -> Loss
            # intermediate pred calculated by passive party  
            passive_model_head_pred = self.vfl_info['passive_predict'][0]  # bs, seq_len, embed_dim
            passive_model_head_attention_mask = self.vfl_info['passive_predict_attention_mask'][0]  # bs, seq_len, embed_dim

            # intermediate pred calculated by active party  
            # if self.args.vfl_model_slice_num == 2:
            #     active_model_body_pred = self.vfl_info['active_predict'][1]  # bs, seq_len, embed_dim
            # active_model_body_pred.requires_grad = True

            # gradient received by active party
            original_dy_dx = self.vfl_info['global_model_body_gradient']  # gradient calculated for local model update
            print(f'original_dy_dx:{type(original_dy_dx)}')
            print(f'{len(original_dy_dx)}')
            
            # for _dy_dx in original_dy_dx:
            #     if _dy_dx != None:
            #         print(_dy_dx.shape)
            #     else:
            #         print(_dy_dx)

            # passive party model tail
            active_model_body = self.vfl_info['active_model_body'].to(self.device)
            active_model_body_params = list(filter(lambda x: x.requires_grad, active_model_body.parameters()))
            print('active_model_body_params:',len(active_model_body_params))

            if self.args.vfl_model_slice_num == 3:
                passive_model_tail = self.vfl_info['local_model_tail'].to(self.device)
                # passive_model_tail.eval()

            # target
            true_label = self.vfl_info['label'].to(self.device)  # CLM: bs, seq_len
            if self.args.model_architect == 'CLM': 
                true_label = true_label[:,-1]
            
            del self.vfl_info

            

            ################ Begin Attack ################
            print(f"sample_count = {true_label.size()[0]}, number of classes = {self.label_size}")
            sample_count = true_label.size()[0]

            recovery_history = []
            recovery_rate_history = []
            rec_rate = 0

            # simulate model forward 
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
            dummy_label = torch.randn(sample_count, self.label_size).to(self.device)
            dummy_label.requires_grad = True
            print(f'dummy_label={dummy_label.shape} true_label={true_label.shape}')
            
            # # set optimizer
            optimizer = torch.optim.Adam([dummy_label],
                        lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  #
            # optimizer = torch.optim.Adam(
            #             itertools.chain([dummy_label], list(passive_model_tail.parameters())),
            #             lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  #
            
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose=False)
            
            start_time = time.time()

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
                    print('Iters',iters,' grad_diff:',grad_diff.item(),' rec_rate:',rec_rate)

                optimizer.step()
                # scheduler.step()
                # print(f"in BLR, i={i}, iter={iters}, time={s_time-e_time}")

                
                if self.early_stop == 1:
                    # if closure().item() < self.early_stop_threshold:
                    #     break
                    if early_stop > self.early_stop_threshold:
                        break

                rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                recovery_rate_history.append(rec_rate)

            end_time = time.time()

            ########## Clean #########
            del (passive_model_head_pred)
            del (passive_model_head_attention_mask)
            del (original_dy_dx)
            del (active_model_body)
            del (dummy_label)

            print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf,time_used=%lf' % (
            sample_count, self.label_size, index, rec_rate, end_time - start_time))

            best_rec_rate = recovery_rate_history[-1]  
            attack_total_time = end_time - start_time
            print(f"BLI, if self.args.apply_defense={self.args.apply_defense}")
            # if self.args.apply_defense == True:
            #     exp_result = f"bs|num_class|attack_party_index|Q|top_trainable|acc,%d|%d|%d|%d|%d|%lf|%s (AttackConfig: %s) (Defense: %s %s)" % (sample_count, self.label_size, index, self.args.Q, self.args.apply_trainable_layer, best_rec_rate, str(self.args.attack_configs), self.args.defense_name, str(self.args.defense_configs))

            # else:
            #     exp_result = f"bs|num_class|attack_party_index|Q|top_trainable|acc,%d|%d|%d|%d|%d|%lf" % (sample_count, self.label_size, index, self.args.Q, self.args.apply_trainable_layer, best_rec_rate)# str(recovery_rate_history)

            # append_exp_res(self.exp_res_path, exp_result)


        print("returning from BLI")
        return best_rec_rate, attack_total_time
        # return recovery_history
