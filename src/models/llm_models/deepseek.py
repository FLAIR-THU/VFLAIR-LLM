"""
copy source codes from transformers, then modify
code based on transformers=4.37.2
"""
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, add_start_docstrings
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, \
    SequenceClassifierOutputWithPast
from transformers.utils import logging, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa


from .third_party_modeling.modeling_deepseek import *
#DeepseekV3ForCausalLM,DeepseekV3PreTrainedModel,DeepseekV3DecoderLayer #DeepseekV3Model
from .third_party_modeling.configuration_deepseek import DeepseekV3Config

from torch.nn import ModuleList, Parameter
from typing import Iterable, Optional, Union, List, Tuple, Callable, Dict, Iterator
from loguru import logger
import torch
import copy
import os
from peft.peft_model import PeftModel
from .base import ModelPartitionPipeline, VFLModel

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

class DeepseekV3ModelSplitter(DeepseekV3Model, VFLModel):
    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self._split_layers(idx_of_layers)

    def _split_layers(self, idx_of_layers: Iterable[int]) -> bool:
        all_decoder_num = len(self.layers)
        
        # print(f'DeepseekV3ModelSplitter _split_layers {list(idx_of_layers)}')
        new_layers = ModuleList()
        for i, layer in enumerate(self.layers):
            if i in idx_of_layers:
                new_layers.append(layer)
            else:
                del layer
        self.layers = new_layers
        
        # update config
        self.config.num_hidden_layers = len(new_layers)
        self.config.all_hidden_layers = all_decoder_num
        self.config.start_layer_idx = idx_of_layers[0]
        self.config.end_layer_idx = idx_of_layers[-1]
        # self.config.n_head = len(new_layers) # n_head = num of attention head
        
        return True

    def _clear_past_key_values(self):
        self.past_key_values = None
    
    
class DeepseekV3ModelHead(DeepseekV3ModelSplitter):
    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self.past_key_values = None
        del self.norm
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

    def _clear_past_key_values(self):
        self.past_key_values = None

    def get_input_embeddings(self):
        if self.embed_tokens != None:
            return self.embed_tokens
        else:
            return None
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # i = 1
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            # print(f'{i} head hidden_states:',hidden_states[0,0,:5])
            # i = i+1
            
        # print('----')
        return {'inputs_embeds': hidden_states,
                'attention_mask': attention_mask}



class DeepseekV3ModelBody(DeepseekV3ModelSplitter,DeepseekV3PreTrainedModel):
    def __init__(self, config: DeepseekV3Config):
        # super().__init__(config)
        DeepseekV3PreTrainedModel.__init__(self, config)
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        
        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
      
        self.past_key_values = None
        del self.norm
        del self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        # no need
        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)

        # if self._use_flash_attention_2:
        #     # 2d mask is passed through the layers
        #     attention_mask = (
        #         attention_mask
        #         if (attention_mask is not None and 0 in attention_mask)
        #         else None
        #     )
        # else:
        #     # 4d mask is passed through the layers
        #     attention_mask = _prepare_4d_causal_attention_mask(
        #         attention_mask,
        #         (batch_size, seq_length),
        #         inputs_embeds,
        #         past_key_values_length,
        #     )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        return {'inputs_embeds': hidden_states,
                'attention_mask': attention_mask}




class DeepseekV3ModelTail(DeepseekV3ModelSplitter,DeepseekV3PreTrainedModel):
    def __init__(self, config: DeepseekV3Config):
        # super().__init__(config)
        DeepseekV3PreTrainedModel.__init__(self, config)
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        
        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
    
        self.past_key_values = None
        # del self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = False #use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        # no need
        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)

        # if self._use_flash_attention_2:
        #     # 2d mask is passed through the layers
        #     attention_mask = (
        #         attention_mask
        #         if (attention_mask is not None and 0 in attention_mask)
        #         else None
        #     )
        # else:
        #     # 4d mask is passed through the layers
        #     attention_mask = _prepare_4d_causal_attention_mask(
        #         attention_mask,
        #         (batch_size, seq_length),
        #         inputs_embeds,
        #         past_key_values_length,
        #     )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        # i = 1
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            # print(f'{i} tail hidden_states:',hidden_states[0,0,:5])
            # i = i+1
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

   


# Global Model Wrapper
class DeepseekV3TailForCausalLM(DeepseekV3ForCausalLM, VFLModel):
    def __init__(self, config: DeepseekV3Config, **kwargs):
        super().__init__(config)
        self.model = DeepseekV3ModelTail(config)
        # Initialize weights and apply final processing
        self.post_init()

    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.model.vfl_split(idx_of_layers)

    def _clear_past_key_values(self):
        self.model._clear_past_key_values()

    @property
    def head_layer(self):
        return self.lm_head

    @head_layer.setter
    def head_layer(self, lm_head):
        self.lm_head = lm_head


class ModelPartitionPipelineDeepseekV3(ModelPartitionPipeline):

    def _load_model_head(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_head = DeepseekV3ModelHead.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            self.all_layer_num = model_head.config.num_hidden_layers
            split_range = range(0, self.split_index[0])
            model_head.vfl_split(split_range)
            # print(list(split_range))
            # print(f'Model Head:{len(model_head.h)} {do_split}')

        return model_head#.to(self.device)

    def _load_model_tail(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        if self.args.model_architect == 'CLM':
            model_tail = DeepseekV3TailForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        # elif self.args.model_architect == 'CLS':
        #     model_tail = DeepseekV3TailForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        # elif self.args.model_architect == 'TQA':
        #     model_tail = DeepseekV3TailForQuestionAnswering.from_pretrained(model_name_or_path, **kwargs)
        else:
            raise ValueError(f"model_architect {self.args.model_architect} not supported for {model_name_or_path}")
        
        if do_split:
            if self.num_of_slices == 2:
                split_range = range(self.split_index[0],model_tail.config.num_hidden_layers)
            else:
                split_range = range(model_tail.config.num_hidden_layers-self.split_index[1],model_tail.config.num_hidden_layers)
            model_tail.vfl_split(split_range)
            # print(list(split_range))
            # print(f'Model Tail:{len(model_tail.model.h)} {do_split}')


        return model_tail#.to(self.device)

    def _load_model_body(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_body = DeepseekV3ModelBody.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            split_range = range(self.split_index[0], model_body.config.num_hidden_layers-self.split_index[1])
            model_body.vfl_split(split_range)
            
            # print(list(split_range))
            # print(f'Model Body:{len(model_body.h)} {do_split}')
           
        
        return model_body#.to(self.device)
