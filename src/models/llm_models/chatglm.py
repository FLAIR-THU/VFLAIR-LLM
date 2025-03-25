"""
copy source codes from transformers, then modify
code based on transformers=4.37.2
"""
from .third_party_modeling.configuration_chatglm import ChatGLMConfig
from .third_party_modeling.modeling_chatglm import *
from .third_party_modeling.tokenization_chatglm import ChatGLMTokenizer

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

from torch.nn import ModuleList, Parameter
from typing import Iterable, Optional, Union, List, Tuple, Callable, Dict, Iterator
from loguru import logger
import torch
import copy
import os
from peft.peft_model import PeftModel
from .base import ModelPartitionPipeline, VFLModel

class ChatGLMModelSplitter(ChatGLMModel, VFLModel):
    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self._split_layers(idx_of_layers)

    def _split_layers(self, idx_of_layers: Iterable[int]) -> bool:
        new_layers = ModuleList()
        for i, layer in enumerate(self.encoder.layers):
            if i in idx_of_layers:
                new_layers.append(layer)
            else:
                del layer
        self.encoder.layers = new_layers

        # update config
        self.config.num_layers = len(new_layers)    
        
        return True

    def _clear_past_key_values(self):
        self.past_key_values = None
    
    
class ChatGLMModelHead(ChatGLMModelSplitter):
    def __init__(self, config: ChatGLMConfig):
        super().__init__(config)
        self.past_key_values = None
        # todo: del norm will cause error when load from original model weight
        # del self.norm

    def _clear_past_key_values(self):
        self.past_key_values = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        self.embedding_output = inputs_embeds # add
        # print('====== head inputs_embeds:',inputs_embeds[0,:2,:5])

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        self.encoder.post_layer_norm = False # no need in non-tail parts
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        return {'inputs_embeds': hidden_states,
                'attention_mask': full_attention_mask,
                'position_ids': position_ids
                }


class ChatGLMModelBody(ChatGLMModelSplitter):
    def __init__(self, config: ChatGLMConfig):
        super().__init__(config)
        self.past_key_values = None
        del self.embedding

        # todo: del norm will cause error when load from original model weight
        # del self.norm

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ## load batch_size, seq_length from inputs_embeds
        # batch_size, seq_length = input_ids.shape
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            # [seq_length, batch_size, embed_dim]
            seq_length, batch_size = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        # if inputs_embeds is None:
        #     inputs_embeds = self.embedding(input_ids)
        # print('====== body inputs_embeds:',inputs_embeds[0,:2,:5])

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        ## load full_attention_mask directly from attention_mask, already processed in local model
        # if full_attention_mask is None:
        #     if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
        #         full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        full_attention_mask = attention_mask
        
        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        self.encoder.post_layer_norm = False # no need in non-tail parts
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        return {'inputs_embeds': hidden_states,
                'attention_mask': full_attention_mask,
                'position_ids': position_ids
                }


class ChatGLMModelTail(ChatGLMModelSplitter):
    def __init__(self, config: ChatGLMConfig):
        super().__init__(config)
        self.past_key_values = None
        # del self.embedding

        # todo: del norm will cause error when load from original model weight
        # del self.norm

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ## load batch_size, seq_length from inputs_embeds
        # batch_size, seq_length = input_ids.shape
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            # [seq_length, batch_size, embed_dim]
            seq_length, batch_size = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        # if inputs_embeds is None:
        #     inputs_embeds = self.embedding(input_ids)
        # print('======== tail inputs_embeds:',inputs_embeds[0,:2,:5])
        
        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        ## load full_attention_mask directly from attention_mask, already processed in local model
        # if full_attention_mask is None:
        #     if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
        #         full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        full_attention_mask = attention_mask
        
        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )



# Global Model Wrapper
class ChatGLMTailForConditionalGeneration(ChatGLMForConditionalGeneration, VFLModel):
    def __init__(self, config: ChatGLMConfig, **kwargs):
        super().__init__(config)
        self.transformer = ChatGLMModelTail(config)
        # Initialize weights and apply final processing
        self.post_init()

    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.transformer.vfl_split(idx_of_layers)

    def _clear_past_key_values(self):
        self.transformer._clear_past_key_values()

    @property
    def head_layer(self):
        return self.lm_head

    @head_layer.setter
    def head_layer(self, lm_head):
        self.lm_head = lm_head

class ChatGLMTailForSequenceClassification(ChatGLMForSequenceClassification, VFLModel):
    def __init__(self, config: ChatGLMConfig, **kwargs):
        super().__init__(config)
        self.transformer = ChatGLMModelTail(config)
        # Initialize weights and apply final processing
        self.post_init()

    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.transformer.vfl_split(idx_of_layers)

    def _clear_past_key_values(self):
        self.transformer._clear_past_key_values()

    @property
    def head_layer(self):
        return self.classifier_head

    @head_layer.setter
    def head_layer(self, classifier_head):
        self.classifier_head = classifier_head


class ModelPartitionPipelineChatGLM(ModelPartitionPipeline):

    def _load_model_head(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_head = ChatGLMModelHead.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            self.all_layer_num = model_head.config.num_layers
            split_range = range(0, self.split_index[0])
            model_head.vfl_split(split_range)
            # print(list(split_range))
            # print(f'Model Head:{len(model_head.h)} {do_split}')

        return model_head#.to(self.device)

    def _load_model_tail(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        if self.args.model_architect == 'CLM':
            model_tail = ChatGLMTailForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        elif self.args.model_architect == 'CLS':
            model_tail = ChatGLMTailForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        # elif self.args.model_architect == 'TQA':
        #     model_tail = ChatGLMTailForQuestionAnswering.from_pretrained(model_name_or_path, **kwargs)
        else:
            raise ValueError(f"model_architect {self.args.model_architect} not supported for {model_name_or_path}")
        
        if do_split:
            if self.num_of_slices == 2:
                split_range = range(self.split_index[0],model_tail.config.num_layers)
            else:
                split_range = range(model_tail.config.num_layers-self.split_index[1],model_tail.config.num_layers)
            model_tail.vfl_split(split_range)
            # print(list(split_range))
            # print(f'Model Tail:{len(model_tail.model.h)} {do_split}')


        return model_tail#.to(self.device)

    def _load_model_body(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_body = ChatGLMModelBody.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            split_range = range(self.split_index[0], model_body.config.num_layers-self.split_index[1])
            model_body.vfl_split(split_range)
            
            # print(list(split_range))
            # print(f'Model Body:{len(model_body.h)} {do_split}')
           
        
        return model_body#.to(self.device)
