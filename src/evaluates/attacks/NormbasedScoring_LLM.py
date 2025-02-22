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
import tensorflow as tf

from evaluates.attacks.attacker import Attacker
from models.global_models import *  # ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from utils.scoring_attack_functions import cosine_similarity
from sklearn.metrics import roc_auc_score


def update_all_norm_leak_auc(norm_leak_auc_dict, grad_list, y):
    for (key, grad) in zip(norm_leak_auc_dict.keys(), grad_list):
        # flatten each example's grad to one-dimensional
        grad = tf.reshape(grad, shape=(grad.shape[0], -1))  #
        # if grad.shape[1] == 1: # the last layer's logit
        #     grad = tf.reshape(grad, shape=[-1])
        #     auc,acc = update_auc(y=y,
        #         predicted_value=grad,
        #         m_auc=norm_leak_auc_dict[key])
        #     #acc = update_acc(y=y,predicted_value=grad)

        # else:
        # auc = update_auc(y=y,
        #             predicted_value= tf.norm(grad, axis=-1, keepdims=False), # torch.norm(grad,dim=-1, keepdim=False), #
        #             m_auc=norm_leak_auc_dict[key])

        ###### auc #######
        # positive_index = tf.where(y>0)
        # negative_index = tf.where(y==0)
        # print(f"positive gradients:{tf.squeeze(tf.gather(grad, positive_index))}")
        # print(f"negative gradients:{tf.squeeze(tf.gather(grad, negative_index))}")

        predicted_value = tf.norm(grad, axis=-1, keepdims=False)
        predicted_value = tf.reshape(predicted_value, shape=(-1))
        if tf.reduce_sum(y) == 0:  # no positive examples in this batch
            return None

        # positive_values = tf.squeeze(tf.gather(predicted_value, positive_index))
        # negative_values = tf.squeeze(tf.gather(predicted_value, negative_index))
        # print(f"positive mean={tf.reduce_mean(positive_values)}, std={tf.math.reduce_std(positive_values)}")
        # print(f"negative mean={tf.reduce_mean(negative_values)}, std={tf.math.reduce_std(negative_values)}")

        val_max = tf.math.reduce_max(predicted_value)
        val_min = tf.math.reduce_min(predicted_value)
        predicted_value = (predicted_value - val_min + 1e-16) / (val_max - val_min + 1e-16)
        predicted_value = predicted_value.numpy()
        y = y.numpy()
        print("y_true:",len(y),y)
        auc = roc_auc_score(y_true=y, y_score=predicted_value)

        ###### acc #######
        predicted_label = np.where(predicted_value > 0.5, 1, 0).reshape(-1)
        # print('predicted_label:',predicted_label[:5])
        # print('y:',y[:5])
        acc = (predicted_label == y).sum() / len(y)

        # print('predicted_label:',predicted_label)
        # print('y:',y)

        return acc, auc


class NormbasedScoring_LLM(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information 
        self.vfl_info = top_vfl.first_epoch_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k
        self.party = args.attack_configs['party']  # parties that launch attacks
        # self.num_run = args.attack_configs['num_run']
        # self.lr = args.attack_configs['lr']
        # self.epochs = args.attack_configs['epochs']
        # self.early_stop = args.attack_configs['early_stop'] if 'early_stop' in args.attack_configs else 0
        # self.early_stop_threshold = args.attack_configs['early_stop_threshold'] if 'early_stop_threshold' in args.attack_configs else 1e-7

        if self.args.model_architect != 'CLM':
            self.label_size = args.num_classes
        else:
            self.label_size = args.model_config.vocab_size
        self.criterion = cross_entropy_for_onehot

        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/NS/'
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
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def attack(self):
        self.set_seed(123)
        for ik in self.party:  # attacker party #ik
            attacked_party = 0
            # index = ik
            # # self.exp_res_dir = self.exp_res_dir + f'{index}/'
            # # if not os.path.exists(self.exp_res_dir):
            # #     os.makedirs(self.exp_res_dir)
            # # self.exp_res_path = self.exp_res_dir + self.file_name

            # # collect necessary information
            true_label = self.vfl_info['label'][attacked_party].to(self.device)  # CLM: bs, seq_len
            print('true_label:', true_label.size())
            sample_count = true_label.size()[0]
            pred_a_gradients_clone = self.vfl_info['global_gradient']

            ################ scoring attack ################
            start_time = time.time()

            tf_pred_a_gradients_clone = tf.convert_to_tensor(pred_a_gradients_clone.cpu().numpy())
            tf_true_label = tf.convert_to_tensor(
                [tf.convert_to_tensor(torch.argmax(true_label[i]).cpu().numpy()) for i in range(len(true_label))])

            print('tf_true_label:', tf_true_label.shape)
            print('tf_pred_a_gradients_clone:', tf_pred_a_gradients_clone.shape)

            # print('pos_idx:',pos_idx)
            # print('tf_pred_a_gradients_clone[pos_idx:pos_idx+1]:',tf_pred_a_gradients_clone[pos_idx:pos_idx+1].shape)

            norm_leak_acc, norm_leak_auc = update_all_norm_leak_auc(
                norm_leak_auc_dict={'only': ''},
                grad_list=[tf_pred_a_gradients_clone],  #
                y=tf_true_label)

            end_time = time.time()

            print(f'batch_size=%d,class_num=%d,acc=%lf,time_used=%lf'
                  % (sample_count, self.label_size, norm_leak_acc, end_time - start_time))

        print("returning from NormbasedScoring_LLM")
        print('norm_leak_acc:', norm_leak_acc)
        return norm_leak_acc, norm_leak_auc
