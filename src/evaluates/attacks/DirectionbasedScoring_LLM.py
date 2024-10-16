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


def update_all_cosine_leak_auc(cosine_leak_auc_dict, grad_list, pos_grad_list, y):
    # grad_list = [ [bs, seq_len, embed_dim]]
    # pos_grad_list = [ [1, seq_len, embed_dim]]

    for (key, grad, pos_grad) in zip(cosine_leak_auc_dict.keys(), grad_list, pos_grad_list):
        # print(f"in cosine leak, [key, grad, pos_grad] = [{key}, {grad}, {pos_grad}]")
        # flatten each example's grad to one-dimensional
        print('grad:',grad.shape) # bs seq_len, embed_dim
        print('pos_grad:',pos_grad.shape) # 1 seq_len, embed_dim

        grad = tf.reshape(grad, shape=(grad.shape[0], -1)) # bs seq_len*embed_dim
        print('reshape grad:',grad.shape)

        # there should only be one positive example's gradient in pos_grad
        pos_grad = tf.reshape(pos_grad, shape=(pos_grad.shape[0], -1)) # 1 seq_len*embed_dim
        print('reshape pos_grad:',pos_grad.shape)

        predicted_value = cosine_similarity(grad, pos_grad).numpy()
        predicted_label = np.where(predicted_value > 0, 1, 0).reshape(-1) # bs
        print('predicted_label:',predicted_label.size)

        _y = y.numpy()
        acc = ((predicted_label == _y).sum() / len(_y))
        # print(f'[debug] grad=[')
        # for _grad, _lable, _pred in zip(grad,y, predicted_label):
        #     print(_grad, _lable, _pred)
        # print("]")

        predicted_value = tf.reshape(predicted_value, shape=(-1))
        if tf.reduce_sum(y) == 0:  # no positive examples in this batch
            return None
        val_max = tf.math.reduce_max(predicted_value)
        val_min = tf.math.reduce_min(predicted_value)
        pred = (predicted_value - val_min + 1e-16) / (val_max - val_min + 1e-16)

        print('true:',_y)
        print('predicted_label:',predicted_label)
        auc = roc_auc_score(y_true=y.numpy(), y_score=pred.numpy())

        print('true:',_y)
        print('predicted_label:',predicted_label)

        return acc, auc


class DirectionbasedScoring_LLM(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information 
        self.top_vfl = top_vfl
        self.vfl_info = top_vfl.first_epoch_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k
        self.party = args.attack_configs['party']  # parties that launch attacks
        if self.args.model_architect != 'CLM':
            self.label_size = args.num_classes
        else:
            self.label_size = args.model_config.vocab_size
        self.criterion = cross_entropy_for_onehot

        self.file_name = 'attack_result.txt'
        self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/DS/'
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
            index = ik

            # collect necessary information
            true_label = self.vfl_info['label'].to(self.device)  # copy.deepcopy(self.gt_one_hot_label)
            print('true_label:', true_label.size()) # bs, num_class

            # batch_data = self.vfl_info['batch_data']
            # loss, train_acc = self.top_vfl.train_batch(batch_data, true_label)
            # pred_a_gradients_clone = copy.deepcopy(self.top_vfl.parties[1].global_gradient)

            pred_a_gradients_clone = self.vfl_info['global_gradient']
            print('pred_a_gradients_clone:', pred_a_gradients_clone.size()) # bs, seq_leb, embed_dim
            del self.vfl_info

            ################ scoring attack ################
            start_time = time.time()
            ################ find a positive gradient ################
            pos_idx = np.random.randint(len(true_label))
            print('pos_idx init:', pos_idx) # 14
            while torch.argmax(true_label[pos_idx]) != torch.tensor(1):
                pos_idx += 1
                if pos_idx >= len(true_label):
                    pos_idx -= len(true_label)
            print('pos_idx after:', pos_idx)# true_label[pos_idx]=1
            ################ found positive gradient ################

            tf_pred_a_gradients_clone = tf.convert_to_tensor(pred_a_gradients_clone.cpu().numpy())
            print('tf_pred_a_gradients_clone:', tf_pred_a_gradients_clone.shape) # bs, seq_leb, embed_dim
            
            tf_true_label = tf.convert_to_tensor(
                [tf.convert_to_tensor(torch.argmax(true_label[i]).cpu().numpy()) for i in range(len(true_label))])
            print('tf_true_label:', tf_true_label.shape) # bs

            cosine_leak_acc, cosine_leak_auc = update_all_cosine_leak_auc(
                cosine_leak_auc_dict={'only': ''},
                grad_list=[tf_pred_a_gradients_clone],# bs, seq_leb, embed_dim
                pos_grad_list=[tf_pred_a_gradients_clone[pos_idx:pos_idx + 1]],  # 1, seq_leb, embed_dim
                y=tf_true_label)

            end_time = time.time()

            print(f'batch_size=%d,class_num=%d,acc=%lf,time_used=%lf'
                  % (self.args.batch_size, self.label_size, cosine_leak_acc, end_time - start_time))

        print("returning from DirectionbasedScoring_LLM")
        return cosine_leak_acc, cosine_leak_auc
