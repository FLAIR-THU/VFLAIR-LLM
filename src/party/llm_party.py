import os
import sys
import numpy as np
import random

sys.path.append(os.pardir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from evaluates.attacks.attack_api import AttackerLoader
from evaluates.defenses.defense_api import DefenderLoader
from load.LoadDataset import load_dataset_per_party, load_dataset_per_party_llm, load_dataset_per_party_backdoor, \
    load_dataset_per_party_noisysample
from load.LoadModels import load_models_per_party_llm
from models.llm_models.base import ModelPartitionPipeline

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
        self.input_attention_mask = {}  # input attention mask type:dict[int,torch.Tensor]
        self.output_tensors = {}  # output embeddings type:dict[int,torch.Tensor]
        self.output_attention_mask = {}  # output attention mask type:dict[int,torch.Tensor]

        self.received_grads = {}  #type:dict[int,torch.Tensor]
        self.models = {}  # type:dict[int,PreTrainedModel]
        self.optimizers = {}  # type:dict[int,torch.optim.Optimizer]
        self.lr_schedulers = {}  # type:dict[int,torch.optim.lr_scheduler.LinearLR]
        self.past_key_values = {}
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
        
        # Load tokenizer and model
        self.tokenizer = None
        self.vis_processors = {}
        self.text_processors = {}
        if need_model:
            self.prepare_model(args, index)

        # Load Data
        if need_data:
            self.prepare_data(args, index)
            self.prepare_data_loader()
            
        self.local_gradient = None
        self.local_pred = None
        self.local_pred_clone = None

        self.origin_pred = None  # for adversarial training

        self.cache = Cache()
        self.prev_batches = []
        self.num_local_updates = 0

        ####### predict results ######
        self.input_shape = None
        self.global_pred = None

        self.local_attention_mask = None  # GPT2
        self.local_sequence_lengths = None  # GPT2 Classification
        self.local_attention_mask = None  # Llama

        ######### defence
        # for adversarial training
        self.adversary_loss = None
        self.mapping_distance = None
        self.head_adversary_loss = None
        self.head_mapping_distance = None
        self.tail_adversary_loss = None
        self.tail_mapping_distance = None


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

    def update_model_data(self, model_data):
        self.args.tokenizer = model_data['tokenizer']
        self.models = model_data['models']
        self.args.config = model_data['config']
        self.args.model_architectures = self.args.config.architectures
        self.args.model_embedded_dim = self.args.config.hidden_size
        if model_data.get('generation_config', None):
            self.args.generation_config = model_data['generation_config']
        # self._set_peft()

    def prepare_data(self, args, index):
        (
            args,
            self.half_dim,
            train_dst,
            test_dst,
        ) = load_dataset_per_party_llm(args, index)

        self.train_data, self.train_label = train_dst
        self.test_data, self.test_label = test_dst
       
    def prepare_data_loader(self, need_auxiliary=0, **kwargs):
        batch_size = self.args.batch_size
        test_batch_size = self.args.test_batch_size
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size, collate_fn=lambda x: x)  # ,
        self.test_loader = DataLoader(self.test_dst, batch_size=test_batch_size, collate_fn=lambda x: x)  # , shuffle=True ,collate_fn=my_collate
        if need_auxiliary == 1 and self.aux_dst != None:
            self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size, collate_fn=lambda x: x)

    def prepare_tokenizer(self, args, model_path):
        # Load Tokenizer
        print('---- Load Tokenizer')

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
    
    def init_Qformer(self,
        num_query_token, vision_width, 
        qformer_hidden_dropout_prob=0.,
        qformer_attention_probs_dropout_prob=0.,
        qformer_drop_path_rate=0.,
    ):
        from models.Qformer import BertConfig, BertLMHeadModel
        encoder_config = BertConfig.from_pretrained("/shared/model/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        encoder_config.hidden_dropout_prob = qformer_hidden_dropout_prob
        encoder_config.attention_probs_dropout_prob = qformer_attention_probs_dropout_prob
        encoder_config.drop_path_list = [x.item() for x in torch.linspace(0, qformer_drop_path_rate, encoder_config.num_hidden_layers)]
        # print(f"Drop_path:{encoder_config.drop_path_list}")
        # print(encoder_config)
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_vision_encoder(self,
        vit_model_name, img_size, drop_path_rate, 
        use_grad_checkpoint, precision, vit_model_path,
        temporal_downsample=True,
        no_lmhra=False, 
        double_lmhra=False,
        lmhra_reduction=2.0, 
        gmhra_layers=8, 
        gmhra_drop_path_rate=0.,
        gmhra_dropout=0.5, 
    ):
        from models.eva_vit import create_eva_vit_g,LayerNorm
        assert vit_model_name == "eva_clip_g", "vit model must be eva_clip_g for current version of VideoChat"
        visual_encoder = create_eva_vit_g(self.args.device,
            img_size, drop_path_rate, 
            use_grad_checkpoint, precision, vit_model_path,
            temporal_downsample=temporal_downsample,
            no_lmhra=no_lmhra, 
            double_lmhra=double_lmhra,
            lmhra_reduction=lmhra_reduction, 
            gmhra_layers=gmhra_layers, 
            gmhra_drop_path_rate=gmhra_drop_path_rate,
            gmhra_dropout=gmhra_dropout, 
        ).to(self.args.device)
        ln_vision = LayerNorm(visual_encoder.num_features).to(self.args.device)
        return visual_encoder, ln_vision
    
    def prepare_processor(self, args):
        print('---- Load processor')
        if args.model_type in ['minicpm','minicpmv']:
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
            
        elif args.model_type == 'videochat':
            vit_configs = args.vit_configs
            vit_model = vit_configs.get("vit_model", "eva_clip_g")
            vit_model_path = vit_configs.get("vit_model_path", None)
            img_size = vit_configs.get("img_size",224)
            drop_path_rate = vit_configs.get("drop_path_rate", 0)
            use_grad_checkpoint = vit_configs.get("use_grad_checkpoint", False)
            vit_precision = vit_configs.get("vit_precision", "fp16")
            freeze_vit = vit_configs.get("freeze_vit", True)
            # uniformerv2
            freeze_mhra = vit_configs.get("freeze_mhra", False)
            temporal_downsample = vit_configs.get("temporal_downsample", True)
            no_lmhra = vit_configs.get("no_lmhra", False)
            double_lmhra = vit_configs.get("double_lmhra", False)
            lmhra_reduction = vit_configs.get("lmhra_reduction", 2.0)
            gmhra_layers = vit_configs.get("gmhra_layers", 8)
            gmhra_drop_path_rate = vit_configs.get("gmhra_drop_path_rate", 0.)
            gmhra_dropout = vit_configs.get("gmhra_dropout", 0.5)
            
            print(f'Loading VIT. Use fp16: {vit_precision}')
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                vit_model, img_size, drop_path_rate, 
                use_grad_checkpoint, vit_precision, vit_model_path,
                temporal_downsample=temporal_downsample,
                no_lmhra=no_lmhra, 
                double_lmhra=double_lmhra,
                lmhra_reduction=lmhra_reduction, 
                gmhra_layers=gmhra_layers, 
                gmhra_drop_path_rate=gmhra_drop_path_rate,
                gmhra_dropout=gmhra_dropout, 
            )
            if freeze_vit:
                print("freeze vision encoder")
                def disabled_train(self, mode=True):
                    """Overwrite model.train with this function to make sure train/eval mode
                    does not change anymore."""
                    return self
                if not freeze_mhra:
                    open_list = []
                    for name, param in self.visual_encoder.named_parameters():
                        if 'mhra' not in name:
                            param.requires_grad = False
                        else:
                            open_list.append(name)
                    # print(f"open module: {open_list}")
                    # print("open ln_vision")
                else:
                    for name, param in self.visual_encoder.named_parameters():
                        param.requires_grad = False
                    self.visual_encoder = self.visual_encoder.eval()
                    self.visual_encoder.train = disabled_train
                    for name, param in self.ln_vision.named_parameters():
                        param.requires_grad = False
                    self.ln_vision = self.ln_vision.eval()
                    self.ln_vision.train = disabled_train
            print('Loading VIT Done')
            
            ###############################################
            qformer_configs = args.qformer_configs
            q_former_model_path = qformer_configs.get("q_former_model_path", None)
            freeze_qformer = qformer_configs.get("freeze_qformer", True)
            num_query_token = qformer_configs.get("num_query_token")
            extra_num_query_token = qformer_configs.get("extra_num_query_token", 64)
            print('num_query_token:',num_query_token)
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features,
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            print('Loading Q-Former')
            if q_former_model_path is not None and os.path.isfile(q_former_model_path):
                checkpoint = torch.load(q_former_model_path, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")
            self.Qformer.load_state_dict(checkpoint, strict=False )
            
            self.Qformer = self.Qformer.to(self.args.device)
            self.query_tokens = self.query_tokens.to(self.args.device)
            # print(f"Add extra {extra_num_query_token} tokens in QFormer")
            self.extra_query_tokens = nn.Parameter(
                torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
            ).to(self.args.device)

            if freeze_qformer:
                print("freeze Qformer")
                for name, param in self.Qformer.named_parameters():
                    param.requires_grad = False
                self.Qformer = self.Qformer.eval()
                self.Qformer.train = disabled_train
                # self.query_tokens.requires_grad = False
                self.query_tokens = self.query_tokens.detach()
                
            print('Loading Q-Former Done')
            
            ##################################
            self.llama_proj = nn.Linear(
                self.Qformer.config.hidden_size, self.local_model.config.hidden_size
            ).to(self.args.device)
            
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
        # Load Tokenizer
        model_path = args.model_path[index]
        self.prepare_tokenizer(args, model_path)

        # Load Model
        result = load_models_per_party_llm(args, index)
        self.models=result['models'] 
        self.full_model_config = result['config'] 
        model_tail_key = args.vfl_model_slice_num - 1
        for _key in self.models.keys(): 
            # update pad token 
            self.models[_key].config.pad_token_id = args.pad_id
            # update generation_config
            if int(_key) == int(model_tail_key):
                self.args.generation_config = self.models[_key].generation_config

        # Load Preprocessor [for multimodaolity tasks]
        if self.index != self.args.k -1 and self.args.task_type == 'MultiModality':
            self.prepare_processor(args)


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
        
    def _set_peft(self):
        """
        peft training or load trained peft weights
        :return:
        """
        if peft_model_path := self.args.model_list[str(self.index)].get('peft_model_path'):
            for i, m in self.models.items():
                _model_path = os.path.join(peft_model_path, f"model_{i}")
                if m and os.path.exists(_model_path):
                    self.models[i] = PeftModel.from_pretrained(m, _model_path).train()

        if _train_conf := vfl_basic_config.vfl_training_config:
            if _train_conf.peft_config:
                self._peft_model_setting()

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
        # print(f"model_{model_index} forward")

        self.input_tensors[model_index] = kwargs.get('inputs_embeds')
        self.input_attention_mask[model_index] = kwargs.get('attention_mask')
        
        self._tensor_to_device(kwargs , self.models[model_index].device)

        resp = self.models[model_index](**kwargs)


        if model_index == self.args.vfl_model_slice_num - 1:
            self.output_tensors[model_index] = resp.get('logits')
            self.output_attention_mask[model_index] = None
        else:
            if resp.get('inputs_embeds') != None:
                self.output_tensors[model_index] = resp.get('inputs_embeds')
            else: # for encoder-decoder inference
                self.output_tensors[model_index] = resp.get('encoder_outputs')['last_hidden_state']

            self.output_attention_mask[model_index] = resp.get('attention_mask')
        return resp 
    
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
            self._tensor_to_device(data_input_dict, self.models[0].device)
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

    def save_pretrained(self,model_folder=None,**kwargs):
        
        if model_folder is None:
            model_folder = get_model_folder(self.args)
            
        for i,m in self.models.items():
            if m:
                if self.args.finetune_name == "LoRA":
                    m = m.merge_and_unload()
                ModelPartitionPipeline.save_pretrained(model_name_or_path=model_folder, 
                                            models={i:m},
                                            **kwargs)