import gc
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Dict, Union
from transformers import PreTrainedModel
import os
from loguru import logger
import torch
from transformers import AutoTokenizer
import copy

class VFLModel(ABC):
    
    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    @abstractmethod
    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        raise NotImplementedError('Not implemented')

    @abstractmethod
    def _clear_past_key_values(self):
        pass


class ModelPartitionPipeline(ABC):
    '''
    split_index: (number of encoders in model head , number of encoders in nudel tail)
    for 2-slice scenario : (n_local, 0)
    for 3-slice scenario : (n_local_head, n_local_tail)
    '''
    def __init__(self, args, all_layer_num, split_index=Union[int, Tuple[int]], is_server=None):
        self.args = args
        self.__split_index = split_index
        self.is_server = is_server
        self.device = args.device
        self.all_layer_num = all_layer_num

    @property
    def num_of_slices(self) -> int:
        '''
        number of model partition slices, 2 or 3
        '''
        return len(self.split_index) + 1

    @property
    def split_index(self) -> Tuple[int]:
        if isinstance(self.__split_index, Tuple):
            if len(self.__split_index) > 2:
                raise ValueError(f"Not supported split_index len:{self.__split_index}")
            return self.__split_index
        elif isinstance(self.__split_index, int):
            return (self.__split_index,)
        else:
            raise ValueError(f"Not supported split_index:{self.__split_index}")

    @property
    def _model_index(self) -> Iterable[int]:
        if self.is_server is None:
            return range(self.num_of_slices) # passive party: 1 or 2 model slice
        elif self.is_server:
            return {1} # active party: 1 model slice
        else:
            idx = set(range(self.num_of_slices))
            idx.remove(1)
            return idx

    def _vfl_model_folder(self, model_path):
        return f"{model_path}_vfl_{self.__split_index}"

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        try:
            return self.from_local_split_model(model_name_or_path, **kwargs)
        except Exception as e:
            logger.warning(f"{repr(e)}\nTry to load from raw model")
            return self._from_raw(model_name_or_path, **kwargs)

    @staticmethod
    def save_pretrained(model_name_or_path: str, models: Dict[int, PreTrainedModel], **kwargs):
        for i, m in models.items():
            m.save_pretrained(os.path.join(model_name_or_path, f"model_{i}"), **kwargs) 

    def from_local_split_model(self, model_name_or_path, model_index=None, **kwargs) -> Dict[int, Union[PreTrainedModel, VFLModel]]:
        """
        try to load from local split model
        :param model_name_or_path:
        :param kwargs:
        :return:
        """
        if model_index == None:
            model_index = self._model_index
        _models = {}
        for i in model_index:
            model_path = os.path.join(model_name_or_path, f"model_{i}")
            if not os.path.exists(model_path):
                # check if vfl model exists
                if os.path.exists(self._vfl_model_folder(model_name_or_path)):
                    logger.info(f"Try existing vfl model: {self._vfl_model_folder(model_name_or_path)}")
                    return self.from_local_split_model(self._vfl_model_folder(model_name_or_path), **kwargs)
                else:
                    raise ValueError(f"Not found required vfl model in {model_name_or_path}")
            if i == 0:
                _model = self._load_model_head(model_path, **kwargs)
            elif i == self.num_of_slices - 1:
                _model = self._load_model_tail(model_path, **kwargs)
            else:
                _model = self._load_model_body(model_path, **kwargs)
            _models.update({i: _model})

        return _models

    def _from_raw(self, model_name_or_path, **kwargs) -> Dict[int, Union[PreTrainedModel, VFLModel]]:
        """
        try to load from raw model locally or remotely
        usually split, save and reload from split model
        :param model_name_or_path:
        :param kwargs:
        :return:
        """
        for i in self._model_index:
            if i == 0:
                _model = self._load_model_head(model_name_or_path, do_split=True, **kwargs)
            elif i == self.num_of_slices - 1:
                _model = self._load_model_tail(model_name_or_path, do_split=True, **kwargs)
            else:
                _model = self._load_model_body(model_name_or_path, do_split=True, **kwargs)

            self.save_pretrained(self._vfl_model_folder(model_name_or_path), models={i: _model})
            del _model
            gc.collect()
            torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        tokenizer.save_pretrained(self._vfl_model_folder(model_name_or_path))
        return self.from_local_split_model(self._vfl_model_folder(model_name_or_path), **kwargs)

    @abstractmethod
    def _load_model_head(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        pass

    @abstractmethod
    def _load_model_tail(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        pass

    @abstractmethod
    def _load_model_body(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        pass

class VFLModelIntermediate(Dict):

    def prepare_for_forward(self,
                            attention_mask=None,
                            past_key_values=None,
                            use_cache=None,
                            position_ids=None,
                            cache_position=None,
                            labels=None):
        """

        :param attention_mask: pass to next
        :param past_key_values: load locally
        :param use_cache: use global setting, uniform for all model split
        :param position_ids: pass to next
        :param cache_position: pass to next
        :param labels: allow for loss computation, only for model[-1]
        :return:
        """
        use_cache=self.get('use_cache') or use_cache

        # if attention_mask is None and (self.get('attention_mask') is None):
        #     # default set attention_mask for CLM generation
        #     attention_mask = torch.ones(self.get('last_hidden_state').shape[:2],
        #                                 device=self.get('last_hidden_state').device)

        ans = {'inputs_embeds': self.get('inputs_embeds'),
               'attention_mask': self.get('attention_mask', attention_mask),
               'use_cache': use_cache, }

        if use_cache:
            ans.update({'past_key_values': past_key_values,
                        # 'output_hidden_states': self.output_hidden_states,
                        'position_ids': self.get('position_ids',position_ids),
                        'cache_position': self.get('cache_position',cache_position) })
        if labels is not None:
            ans.update({'labels': labels})
        return ans

    def to(self, device):
        for v in self.__dict__.values():
            if isinstance(v, torch.Tensor):
                v.to(device)
        return self

    def to_json(self):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__.update({k: v.tolist()})

    def detach(self):
        new_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                v_new = v.detach().clone()
                v_new.requires_grad = v.requires_grad
                new_dict.update({k: v_new})
            else:
                new_dict.update({k: copy.deepcopy(v)})
        return self.__class__(**new_dict)

    def get(self, key, default=None):
        if key == 'inputs_embeds':
            return super().get(key, super().get('last_hidden_state'))
        return super().get(key, default)

