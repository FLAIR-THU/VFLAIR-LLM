import os
import sys
import numpy as np
import random

sys.path.append(os.pardir)

import torch
from torch.utils.data import DataLoader
from loguru import logger

from evaluates.attacks.attack_api import AttackerLoader
from evaluates.defenses.defense_api import DefenderLoader
from load.LoadDataset import load_dataset_per_party, load_dataset_per_party_llm, load_dataset_per_party_backdoor, \
    load_dataset_per_party_noisysample
from load.LoadModels import load_models_per_party_llm

from utils import timer
from utils.noisy_label_functions import add_noise
from utils.noisy_sample_functions import noisy_sample
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor, pairwise_dist
from utils.communication_protocol_funcs import Cache
from .party_utils import get_model_folder
import torch
import re
import collections
from transformers import PreTrainedModel, AutoTokenizer
from peft import get_peft_model,PeftModel
from config import vfl_basic_config
np_str_obj_array_pattern = re.compile(r'[SaUO]')

from models.llm_models.processors.blip_processors import *

PROCESSOR_DICT = {
    'Blip2ImageTrainProcessor':Blip2ImageTrainProcessor,
    'Blip2ImageEvalProcessor':Blip2ImageEvalProcessor,
    'BlipCaptionProcessor': BlipCaptionProcessor,

    'MiniCPM_Transform': MiniCPM_Transform,

}

class Party(object):
    def __init__(self, args, index, need_data=True, need_model=True):
        self.name = "party#" + str(index + 1)
        self.index = index
        self.args = args
        args.need_auxiliary = 0
        args.dataset = args.dataset_split['dataset_name']
        # data for training and testing
        self.half_dim = -1
        self.train_data = None
        self.test_data = None
        self.aux_data = None
        self.train_label = None
        self.test_label = None
        self.aux_label = None
        self.train_attribute = None
        self.test_attribute = None
        self.aux_attribute = None
        self.train_dst = None
        self.test_dst = None
        self.aux_dst = None
        self.train_loader = None
        self.test_loader = None
        self.aux_loader = None
        self.attribute_loader = None
        self.attribute_iter = None
        self.local_batch_data = None
        self.local_batch_attention_mask = None
        self.local_batch_token_type_ids = None
        # backdoor poison data and label and target images list
        self.train_poison_data = None
        self.train_poison_label = None
        self.test_poison_data = None
        self.test_poison_label = None
        self.train_target_list = None
        self.test_target_list = None
        # store attributes of model slices
        self.input_tensors = {}  # input intermediate type:dict[int,torch.Tensor]
        self.output_tensors = {}  # output embeddings type:dict[int,torch.Tensor]
        self.output_attention_mask = {}  # output attention mask type:dict[int,torch.Tensor]

        self.received_grads = {}  #type:dict[int,torch.Tensor]
        self.models = {}  # type:dict[int,PreTrainedModel]
        self.optimizers = {}  # type:dict[int,torch.optim.Optimizer]
        self.lr_schedulers = {}  # type:dict[int,torch.optim.lr_scheduler.LinearLR]

        self.is_first_forward_iter = 1

        # local model
        self.local_model = None
        self.local_model_optimizer = None
        # local model tail (3-slice scenario)
        self.local_model_tail = None
        self.local_model_tail_optimizer = None
        # global_model
        self.global_model = None
        self.global_model_optimizer = None
        
        # tokenizer
        self.tokenizer = None
        self.vis_processors = {}
        self.text_processors = {}

        if need_model:
            self.prepare_model(args, index)
        
        # attack and defense
        # self.attacker = None
        self.defender = None

        # Data
        if need_data:
            self.prepare_data(args, index)
        # self.prepare_attacker(args, index)
        # self.prepare_defender(args, index)

        self.local_gradient = None
        self.local_pred = None
        self.local_pred_clone = None

        self.origin_pred = None  # for adversarial training

        self.cache = Cache()
        self.prev_batches = []
        self.num_local_updates = 0

        self.party_time = 0

        ####### predict results ######
        self.input_shape = None
        self.global_pred = None

        self.local_attention_mask = None  # GPT2
        self.local_sequence_lengths = None  # GPT2 Classification
        self.local_attention_mask = None  # Llama

        # for adversarial training
        self.adversary_loss = None
        self.mapping_distance = None

    def set_is_first_forward_iter(self, value):
        self.is_first_forward_iter = value

    @property
    def is_active_party(self):
        return self.index == self.args.k - 1

    def eval(self):
        for m in self.models.values():
            if m:
                m.eval()

    def train(self,*args,**kwargs):
        for m in self.models.values():
            if m:
                m.train(*args,**kwargs)

    def prepare_data(self, args, index):
        (
            args,
            self.half_dim,
            train_dst,
            test_dst,
        ) = load_dataset_per_party_llm(args, index)

        self.train_data, self.train_label = train_dst
        self.test_data, self.test_label = test_dst
        # self.train_data, self.train_label = train_dst[0][:20],train_dst[1][:20]
        # self.test_data, self.test_label = train_dst[0][:20],train_dst[1][:20]

    def prepare_data_loader(self, need_auxiliary=0, **kwargs):
        # self.train_loader = DataLoader(self.train_dst, batch_size=batch_size) # , 
        # self.test_loader = DataLoader(self.test_dst, batch_size=batch_size) # , shuffle=True ,collate_fn=my_collate
        # if self.args.need_auxiliary == 1 and self.aux_dst != None:
        #     self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size)

        batch_size = self.args.batch_size
        test_batch_size = self.args.test_batch_size
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size, collate_fn=lambda x: x)  # ,
        self.test_loader = DataLoader(self.test_dst, batch_size=test_batch_size,
                                      collate_fn=lambda x: x)  # , shuffle=True ,collate_fn=my_collate
        if need_auxiliary == 1 and self.aux_dst != None:
            self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size, collate_fn=lambda x: x)

    def prepare_tokenizer(self, args, model_path):
        # Load Tokenizer
        print('--- Load Tokenizer')

        self.args.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.args.tokenizer.padding_side = args.padding_side if (args.padding_side in ["left", "right"]) else "left"
        
        if self.args.pad_token == "default":
            print('default tokenizer.pad_token:',self.args.tokenizer.pad_token)
            if self.args.tokenizer.pad_token is None:
                self.args.tokenizer.pad_token = args.tokenizer.eos_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
                self.args.pad_id = args.tokenizer.convert_tokens_to_ids(args.tokenizer.eos_token)  #
            else:
                self.args.pad_token = args.tokenizer.pad_token
                self.args.pad_id = args.tokenizer.convert_tokens_to_ids(self.args.pad_token)  #
        else:
            self.args.pad_id = args.tokenizer.convert_tokens_to_ids(self.args.pad_token)  #
            if self.args.pad_id != None: 
                self.args.tokenizer.pad_token = self.args.pad_token  # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
            else: # invalid pad token set
                print('invalid pad token set, use default pad token from the tokenizer')
                self.args.pad_token = self.args.tokenizer.pad_token
                self.args.pad_id = args.tokenizer.convert_tokens_to_ids(self.args.pad_token) 

        print('pad_token:',self.args.pad_token,'  pad_id:',self.args.pad_id)

        self.tokenizer = args.tokenizer

    def prepare_processor(self, args):
        print('---- Load processor')
        vis_proc_cfg = args.vis_processor_config
        txt_proc_cfg = args.text_processor_config 

        if vis_proc_cfg is not None:
            vis_train_cfg = vis_proc_cfg.get("train")
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["train"] = self._build_proc_from_cfg(vis_train_cfg)
            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

        print('vis_processors:',self.vis_processors.keys())
        print('text_processors:',self.text_processors.keys())

        return
    
    @staticmethod
    def _build_proc_from_cfg(cfg):
        # print('cfg:',cfg)
        return (
            # registry.get_processor_class(cfg.name).from_config(cfg)
            PROCESSOR_DICT[cfg['name']].from_config(cfg)
            if cfg is not None
            else None
        )

        
    def prepare_optimizer(self,args):
        print('---- Load optimizer')
        # Optimizer
        self.optimizers = {}
        for i in self.models.keys():
            trainable_params = list(filter(lambda x: x.requires_grad, self.models[i].parameters()))
            if len(trainable_params) > 0:
                optimizer = torch.optim.Adam(trainable_params, lr=args.main_lr)
            else:
                optimizer = None
            self.optimizers.update({i: optimizer})
        print(f'Party {self.index} Optimizer:',self.optimizers.keys())
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor=0.01, total_iters=10)
        # self.lr_schedulers.update({i: scheduler})

    def prepare_model(self, args, index):
        print(f'=== prepare_model for Party {index} ===')
        # Load Tokenizer
        model_path = args.model_path[index]
        self.prepare_tokenizer(args, model_path)

        # Load Preprocessor [for multimodaolity tasks]
        if self.index != self.args.k -1 and self.args.task_type == 'MultiModality':
            self.prepare_processor(args)
        
        # Load Model
        result = load_models_per_party_llm(args, index)
        self.models=result['models'] #.update(result['models'])
        

        model_tail_key = args.vfl_model_slice_num - 1
        for _key in self.models.keys(): 
            # update pad token configs
            self.models[_key].config.pad_token_id = args.pad_id
            
            # update generation_config
            if int(_key) == int(model_tail_key):
                self.args.generation_config = self.models[_key].generation_config

        # self.args.generation_config = result['generation_config'] 

        self.args.model_config = result['config']
        self.args.model_dtype = result['model_dtype']
        self.args.config = result['config'] # model config
        self.args.config.pad_token_id = args.pad_id
        self.args.model_architectures = result['model_architectures'] 
        self.args.model_embedded_dim = result['model_embedded_dim'] 
        self.args.all_encoders_num = result['all_encoders_num'] 
        self.args.global_encoders_num = self.args.all_encoders_num - args.local_encoders_num - args.local_tail_encoders_num
        print(f'model slices:',self.models.keys())
        if args.vfl_model_slice_num==3:
            print(f'model partition: 0head-{args.local_encoders_num}/1body-{args.global_encoders_num}/2tail-{args.local_tail_encoders_num}')
        else:
            print(f'model partition: 0head-{args.local_encoders_num}/1tail-{args.global_encoders_num}')

        # Load Optimizer
        self.prepare_optimizer(args)
        
       
    def _peft_model_setting(self):
        _train_conf = vfl_basic_config.vfl_training_config
        logger.info(f"enable peft model setting: \n{str(_train_conf.peft_config)}")

        for i in _train_conf.trainable_slice:
            model = self.models.get(i,None)
            if model:
                model.enable_input_require_grads()
                if not isinstance(model,PeftModel):
                    peft_model = get_peft_model(model, _train_conf.peft_config)
                    peft_model.print_trainable_parameters()
                    self.models.update({i: peft_model})
                else:
                    model.set_adapter('default')
                trainable_params = filter(lambda x: x.requires_grad, model.parameters())
                # 定义优化器和学习率调度器
                optimizer = torch.optim.AdamW(trainable_params, lr=_train_conf.training_args.learning_rate)
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor=0.01, total_iters=10)
                self.optimizers.update({i: optimizer})
                self.lr_schedulers.update({i: scheduler})

    def label_to_one_hot(self, target, num_classes=10):
        target = target.long()
        # print('label_to_one_hot:', target, type(target),type(target[0]))
        try:
            _ = target.size()[1]
            # print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            # print("use unsqueezed target", target.size(),type(target))

            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def receive_gradient(self, gradient):
        self.local_gradient = gradient
        return

    def _tensor_to_device(self, dict_like:dict, device):
        for k,v in dict_like.items():
            if isinstance(v,torch.Tensor):
                dict_like[k] = v.to(device)

    @timer()
    def forward(self, model_index, **kwargs):
        logger.debug(f"model_{model_index} forward")

        self.input_tensors[model_index] = kwargs.get('inputs_embeds')
        
        self._tensor_to_device(kwargs , self.models[model_index].device)

        resp = self.models[model_index](**kwargs)

        if model_index == self.args.vfl_model_slice_num - 1:
            self.output_tensors[model_index] = resp.get('logits')
            self.output_attention_mask[model_index] = None
        else:
            self.output_tensors[model_index] = resp.get('inputs_embeds')
            self.output_attention_mask[model_index] = resp.get('attention_mask')
        # self.output_tensors[model_index].requires_grad = True
        return resp #self._detach_tensor(resp)

    def give_current_lr(self):
        return (self.local_model_optimizer.state_dict()['param_groups'][0]['lr'])

    def LR_decay(self, i_epoch):
        eta_0 = self.args.main_lr
        eta_t = eta_0 / (np.sqrt(i_epoch + 1))
        if self.local_model_optimizer != None:
            for param_group in self.local_model_optimizer.param_groups:
                param_group['lr'] = eta_t
        if self.local_model_tail_optimizer != None:
            for param_group in self.local_model_tail_optimizer.param_groups:
                param_group['lr'] = eta_t

    def obtain_local_data(self, data_input_dict, **kwargs):
        if data_input_dict:
            self._tensor_to_device(data_input_dict,self.models[0].device)
            self.local_data_input = data_input_dict
        else:
            pass

    @property
    def local_model(self):
        if 0 in self.models:
            return self.models[0]
        else:
            return None

    @local_model.setter
    def local_model(self, model):
        if model is None:
            pass
        else:
            self.models.update({0: model})

    @property
    def local_model_tail(self):
        if 2 in self.models:
            return self.models[2]
        else:
            return None

    @local_model_tail.setter
    def local_model_tail(self, model):
        if model is None:
            pass
        else:
            self.models.update({2: model})


    @property
    def local_model_optimizer(self):
        if 0 in self.optimizers:
            return self.optimizers[0]
        else:
            return None

    @local_model_optimizer.setter
    def local_model_optimizer(self, optimizer):
        if optimizer is None:
            pass
        else:
            self.optimizers.update({0: optimizer})

    @property
    def local_model_tail_optimizer(self):
        if 2 in self.optimizers:
            return self.optimizers[2]
        else:
            return None

    @local_model_tail_optimizer.setter
    def local_model_tail_optimizer(self, optimizer):
        if optimizer is None:
            pass
        else:
            self.optimizers.update({2: optimizer})


    @property
    def local_pred(self):
        return self.output_tensors[0]

    @local_pred.setter
    def local_pred(self, tensor):
        self.output_tensors[0]=tensor

    @property
    def global_model(self):
        if 1 in self.models:
            return self.models[1]
        else:
            return None

    @global_model.setter
    def global_model(self, model):
        if model is None:
            pass
        else:
            self.models.update({1: model})

    @property
    def global_model_optimizer(self):
        return self.optimizers[1]

    @global_model_optimizer.setter
    def global_model_optimizer(self, optimizer):
        self.optimizers.update({1: optimizer})


    def local_forward(self):
        # args.local_model()
        pass


    def _detach_tensor(self, dict_like: dict):
        return dict_like  # todo: need to check whether used in local mode
        # for key, value in dict_like.items():
        #     if isinstance(value, torch.Tensor) and value.requires_grad:
        #         dict_like[key] = value.detach().clone()
        #         dict_like[key].requires_grad = True
        # return dict_like

    # def backward(self, model_index, **kwargs):
    #     logger.debug(f"model_{model_index} backward")
    #     self.output_tensors[model_index]['inputs_embeds'].backward(**kwargs)

    def optimizer_step(self, model_index):
        logger.debug(f"model_{model_index} optimize")
        self.optimizers[model_index].step()
        self.optimizers[model_index].zero_grad()

    # def save_pretrained(self,model_index,model_id,model_folder=None,**kwargs):
    #     if model_folder is None:
    #         model_folder = get_model_folder()
    #     for i,m in self.models.items():
    #         if m and i in model_index:
    #             logger.debug(f"save model {i}")
    #             ModelPartitionPipeline.save_pretrained(model_name_or_path=os.path.join(*filter(lambda x:x is not None,[model_folder,model_id])),
    #                                         models={i:m},
    #                                         **kwargs)