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
from random import randint

from random import randint
from evaluates.attacks.attacker import Attacker
from models.global_models import *  # ClassificationModelHostHead, ClassificationModelHostTrainableHead
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res


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


class DirectLabelScoring_LLM(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.vfl_info = top_vfl.first_epoch_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.sample_count = args.batch_size
        self.k = args.k
        self.party = args.attack_configs['party']  # parties that launch attacks
        if self.args.model_architect != 'CLM':
            self.label_size = args.num_classes
        else:
            self.label_size = args.model_config.vocab_size


        self.criterion = cross_entropy_for_onehot

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

            true_label = self.vfl_info['label'].to(self.device)  # copy.deepcopy(self.gt_one_hot_label)
            print('true_label:',true_label.shape)
            global_gradient = self.vfl_info['global_gradient']
            print('global_gradient:',global_gradient.shape)

            del self.vfl_info
            start_time = time.time()

            pred_label = []
            for _gradient in global_gradient:
                pred_idx = -1
                for idx in range(len(_gradient)):
                    if _gradient[idx] < 0.0:
                        pred_idx = idx
                        break
                if pred_idx == -1:
                    pred_idx = randint(0, self.num_classes - 1)
                pred_label.append(pred_idx)

            one_hot_pred_label = label_to_one_hot(torch.tensor(pred_label), self.num_classes).to(self.device)
            rec_rate = self.calc_label_recovery_rate(one_hot_pred_label, true_label)

            end_time = time.time()
            attack_total_time = end_time - start_time


            print(f"DLI, if self.args.apply_defense={self.args.apply_defense}")
            print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf' % \
                  (self.sample_count, self.label_size, index, rec_rate))

        print("returning from DLI")
        return rec_rate,attack_total_time
        # return recovery_history
