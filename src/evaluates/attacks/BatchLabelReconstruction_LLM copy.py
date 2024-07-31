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

    def cal_loss(self, pred, real_label):
        if self.args.model_architect == 'CLM': # label: bs, vocab_size
            lm_logits = pred.logits  # [bs, seq_len, vocab_size]
            next_token_logits = lm_logits[:, -1, :]
            # print(f'cal loss next_token_logits={next_token_logits.shape} real_label={real_label.shape}')
            loss = self.criterion(next_token_logits, real_label)
        elif self.args.model_architect == 'CLS': # label: bs, num_labels
            pooled_logits = pred.logits # [bs, num_labels]
            real_label = torch.argmax(real_label,-1) # [bs, num_labels] -> [bs]
            # print('pooled_logits.view(-1, self.num_labels):',pooled_logits.view(-1, self.num_labels).shape)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(pooled_logits.view(-1, self.label_size), real_label.view(-1))
        
        return loss

    def attack(self):
        self.set_seed(123)


        for attacker_ik in self.party: # attacker party #attacker_ik
            assert attacker_ik == (self.k - 1), 'Only Active party launch input inference attack'
            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)
            index = attacker_ik


            ####### collect necessary information #######
            # intermediate pred calculated by active party  
            active_model_body_pred = self.vfl_info['active_predict'][1]  # bs, seq_len, embed_dim
            active_model_body_attention_mask = self.vfl_info['active_predict_attention_mask'][1]  
            active_model_body_pred.requires_grad = True

            passive_model_tail_pred = self.vfl_info['passive_predict'][2]  
            print('passive_model_tail_pred:',passive_model_tail_pred[:5])

            # gradient received by active party
            global_gradient = self.vfl_info['global_gradient'] # bs, seq_len, embed_dim
            print(f'global_gradient:{global_gradient.shape} {global_gradient[0,0,:5]}')
            

            # passive party model tail
            local_model_tail = self.vfl_info['local_model_tail'].to(self.device)
            # local_model_tail.eval()

            dummy_local_model_tail = copy.deepcopy(local_model_tail).to(self.device)
            for name, m in dummy_local_model_tail.named_parameters():
                print(f'{name} {type(m)} original:',m[0])
                torch.nn.init.zeros_(m)
                print(f'{name} after:',m[0])

                    

            # target
            test_data = self.vfl_info["test_data"][0] 
            test_label = self.vfl_info["test_label"][0] 
            
            if len(test_data) > self.attack_sample_num:
                test_data = test_data[:self.attack_sample_num]
                test_label = test_label[:self.attack_sample_num]
                # attack_test_dataset = attack_test_dataset[:self.attack_sample_num]
            
            if self.args.dataset == 'Lambada':
                attack_test_dataset = LambadaDataset_LLM(self.args, test_data, test_label, 'test')
            else:
                attack_test_dataset = PassiveDataset_LLM(self.args, test_data, test_label)

            attack_info = f'Attack Sample Num:{len(attack_test_dataset)}'
            print(attack_info)
            append_exp_res(self.args.exp_res_path, attack_info)

            # true_label = self.vfl_info['label'].to(self.device)  # CLM: bs, seq_len
            # if self.args.model_architect == 'CLM': 
            #     true_label = true_label[:,-1]

            test_data_loader = DataLoader(attack_test_dataset, batch_size=batch_size ,collate_fn=lambda x:x ) # ,
            del(self.vfl_info)
            # test_data_loader = self.vfl_info["test_loader"][0] # Only Passive party has origin input
            
            flag = 0
            enter_time = time.time()
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

                # real received intermediate result
                self.top_vfl.parties[0].obtain_local_data(data_inputs)
                self.top_vfl.parties[0].gt_one_hot_label = batch_label

                all_pred_list = self.top_vfl.pred_transmit()
                real_results = all_pred_list[0]
                self.top_vfl._clear_past_key_values()



            ################ Begin Attack ################
            print(f"sample_count = {active_model_body_pred.size()[0]}, number of classes = {self.label_size}")
            sample_count = active_model_body_pred.size()[0]

            recovery_history = []
            recovery_rate_history = []
            rec_rate = 0

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

                    # pred/loss using fake top model/fake label
                    # print('model tail input active_model_body_pred:',active_model_body_pred[0,0,:5])
                    # print('model tail input active_model_body_attention_mask:',active_model_body_attention_mask[0,:5])
                    # print('local_model_tail:',local_model_tail.head_layer.weight[0,:5])
                    
                    dummy_local_model_tail_pred = local_model_tail(
                        inputs_embeds = active_model_body_pred,
                        attention_mask= active_model_body_attention_mask)
                    print('dummy_local_model_tail_pred logits:',dummy_local_model_tail_pred.logits[:5])
                    print('passive_model_tail_pred logits:',passive_model_tail_pred[:5])


                    

                    dummy_loss = self.cal_loss(dummy_local_model_tail_pred, dummy_label)
                    dummy_gradient = torch.autograd.grad(dummy_loss, active_model_body_pred, 
                        create_graph=True, retain_graph=True)[0] # 4, 256, 768
                    print(f'dummy_loss:{dummy_loss} dummy_label:{dummy_label[:2]}')
                    print(f'dummy_gradient:{dummy_gradient[0,0,:5]}')

                    real_loss = self.cal_loss(dummy_local_model_tail_pred, true_label)
                    real_gradient = torch.autograd.grad(real_loss, active_model_body_pred, 
                        create_graph=True, retain_graph=True)[0] # 4, 256, 768
                    print(f'real_loss:{real_loss} true_label:{true_label[:2]}')
                    print(f'real_gradient:{real_gradient[0,0,:5]}')

                    assert 1>2

                    
                    grad_diff = ((dummy_gradient-global_gradient) ** 2).mean()
                    grad_diff.backward(retain_graph=True)

                    # local_model_tail_params = []
                    # for name, param in local_model_tail.named_parameters():
                    #     if param.requires_grad:
                    #         weight = torch.autograd.grad(dummy_loss, local_model_tail_params, 
                    #     retain_graph=True, allow_unused=True)
                    #         print(f'{name} {param.shape}')
                                                    
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
                    
                    

                    if iters%2==0:
                        print('Iters',iters,' grad_diff:',grad_diff.item(),' rec_rate:',rec_rate)
                        assert 1>2
                    return grad_diff

                optimizer.step(closure)
                e_time = time.time()
                # print(f"in BLR, i={i}, iter={iters}, time={s_time-e_time}")

                if self.early_stop == 1:
                    if closure().item() < self.early_stop_threshold:
                        break

                rec_rate = self.calc_label_recovery_rate(dummy_label, true_label)
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
