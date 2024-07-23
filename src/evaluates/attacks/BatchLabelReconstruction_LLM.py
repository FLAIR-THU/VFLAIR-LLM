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
from models.global_models import *  # ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
torch.backends.cudnn.enabled = False


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
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == gt_label).item()
        total = dummy_label.shape[0]
        return success / total

    def cal_loss(self, pred, real_label):
        if self.args.model_architect == 'CLM':
            lm_logits = pred.logits  # [bs, seq_len, vocab_size]
            next_token_logits = lm_logits[:, -1, :]
            # print(f'cal loss next_token_logits={next_token_logits.shape} real_label={real_label.shape}')
            loss = self.criterion(next_token_logits, real_label)
        return loss

    def attack(self):
        self.set_seed(123)


        for attacker_ik in self.party: # attacker party #attacker_ik
            assert attacker_ik == (self.k - 1), 'Only Active party launch input inference attack'
            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)
            index = attacker_ik


            ####### collect necessary information #######
            # passive_model_head_pred = self.vfl_info['passive_predict'][0]  
            # passive_model_tail_pred = self.vfl_info['passive_predict'][2]  
            active_model_body_pred = self.vfl_info['active_predict'][1]  # bs, seq_len, embed_dim
            active_model_body_attention_mask = self.vfl_info['active_predict_attention_mask'][1]  
            active_model_body_pred.requires_grad = True

            # gradient received by active party
            global_gradient = self.vfl_info['global_gradient'] # bs, seq_len, embed_dim
            print(f'global_gradient:{type(global_gradient)} {global_gradient.shape}')
            
            # original_dy_dx = self.vfl_info['global_model_body_gradient']  # gradient calculated for local model update
            # print(f'original_dy_dx:{type(original_dy_dx)} {len(original_dy_dx)}')
            # for _dy_dx in original_dy_dx:
            #     if _dy_dx != None:
            #         print(_dy_dx.shape)
            #     else:
            #         print(_dy_dx)

            local_model_tail = self.vfl_info['local_model_tail'].to(self.device)
            local_model_tail.eval()

            true_label = self.vfl_info['label'].to(self.device)  # CLM: bs, seq_len
            if self.args.model_architect == 'CLM': 
                true_label = true_label[:,-1]

            print(f"sample_count = {active_model_body_pred.size()[0]}, number of classes = {self.label_size}")
            sample_count = active_model_body_pred.size()[0]

            recovery_history = []
            recovery_rate_history = []

            # set fake label
            dummy_label = torch.randn(sample_count, self.label_size).to(self.device)
            dummy_label.requires_grad = True
            print(f'dummy_label={dummy_label.shape} true_label={true_label.shape}')

            optimizer = torch.optim.Adam(
                [dummy_label],
                lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)  #

            # === Begin Attack ===
            start_time = time.time()

            for iters in range(1, self.epochs + 1):
                # s_time = time.time()

                def closure():
                    optimizer.zero_grad()

                    # fake pred/loss using fake top model/fake label
                    dummy_local_model_tail_pred = local_model_tail(
                        inputs_embeds = active_model_body_pred,
                        attention_mask=active_model_body_attention_mask)

                    dummy_loss = self.cal_loss(dummy_local_model_tail_pred, dummy_label)

                    # local_model_tail_params = []
                    # for param in local_model_tail.parameters():
                    #     if param.requires_grad:
                    #         local_model_tail_params.append(param)
                    # dummy_dy_dx_a = torch.autograd.grad(dummy_loss, local_model_tail_params, 
                    #     retain_graph=True, allow_unused=True)
                    # print(f'dummy_dy_dx_a:{type(dummy_dy_dx_a)} {len(dummy_dy_dx_a)}')
                    # for _dy_dx in dummy_dy_dx_a:
                    #     if _dy_dx != None:
                    #         print(_dy_dx.shape)
                    #     else:
                    #         print(_dy_dx)

                    # grad_diff = 0
                    # for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                    #     grad_diff += ((gx - gy) ** 2).sum()
                    # grad_diff.backward(retain_graph=True)
                    
                    dummy_gradient = torch.autograd.grad(dummy_loss, active_model_body_pred, 
                        create_graph=True, retain_graph=True)[0] # 4, 256, 768
                    
                    # weight_grad = torch.autograd.grad(dummy_gradient, dummy_label, retain_graph=True, allow_unused=True) # 4, 256, 768
                    # print(f'dummy_gradient weight_grad:',weight_grad)

                    # print(f'dummy_gradient:{dummy_gradient.shape}  global_gradient:{global_gradient.shape}')
                    grad_diff = ((dummy_gradient-global_gradient) ** 2).mean()
                    grad_diff.backward(retain_graph=True)

                    if iters%20==0:
                        print('Iters',iters,' grad_diff:',grad_diff.item())
                    return grad_diff

                optimizer.step(closure)
                e_time = time.time()
                # print(f"in BLR, i={i}, iter={iters}, time={s_time-e_time}")

                if self.early_stop == 1:
                    if closure().item() < self.early_stop_threshold:
                        break

                rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
                # if iters%100==0:
                #     print('Iters',iters,' rec_rate:',rec_rate)
                recovery_rate_history.append(rec_rate)
                end_time = time.time()

            ########## Clean #########
            # del (local_model_tail)
            # del (dummy_pred_b)
            # del (dummy_label)

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
