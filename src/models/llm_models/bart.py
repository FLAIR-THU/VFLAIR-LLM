# """
# copy source codes from transformers, then modify
# code based on transformers=4.37.2
# """
# import inspect
# from torch.nn import ModuleList, Parameter
# from typing import Iterable, Optional, Union, List, Tuple, Callable, Dict, Iterator
# from loguru import logger
# import torch
# import copy
# import os
# from peft.peft_model import PeftModel
# from .base import ModelPartitionPipeline, VFLModel
# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.models.bart.modeling_bart import *


# class BartEncoderModelSplitter(BartEncoder, VFLModel):
#     def vfl_split(self, idx_of_layers: Iterable[int], is_encoder=1) -> bool:
#         return self._split_layers(idx_of_layers, is_encoder)

#     def _split_layers(self, idx_of_layers: Iterable[int], is_encoder) -> bool:
#         print(f'BartEncoder _split_layers {list(idx_of_layers)}')
        
#         origin_len = len(self.layers)

#         new_layers = ModuleList()
#         for i, layer in enumerate(self.layers):
#             if i in idx_of_layers:
#                 new_layers.append(layer)

#         self.layers = new_layers
        
#         # update config
#         self.config.encoder_layers = len(new_layers)
        
#         return True

#     def _clear_past_key_values(self):
#         self.past_key_values = None
        

# class BartDecoderModelSplitter(BartDecoder, VFLModel):
#     def vfl_split(self, idx_of_layers: Iterable[int], is_encoder=1) -> bool:
#         return self._split_layers(idx_of_layers, is_encoder)

#     def _split_layers(self, idx_of_layers: Iterable[int], is_encoder) -> bool:
#         print(f'BartDecoder _split_layers {list(idx_of_layers)}')
        
#         origin_len = len(self.layers)

#         new_layers = ModuleList()
#         for i, layer in enumerate(self.layers):
#             if i in idx_of_layers:
#                 new_layers.append(layer)

#         self.layers = new_layers
        
#         # update config
#         self.config.decoder_layers = len(new_layers)
        
#         return True

#     def _clear_past_key_values(self):
#         self.past_key_values = None


 
    
# class BartEncoderHead(BartEncoderModelSplitter):
#     def __init__(self, config: BartConfig):
#         super().__init__(config)

#     def _clear_past_key_values(self):
#         self.past_key_values = None

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutput]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input = input_ids
#             input_ids = input_ids.view(-1, input_ids.shape[-1])
#         elif inputs_embeds is not None:
#             input = inputs_embeds[:, :, -1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

#         embed_pos = self.embed_positions(input)
#         embed_pos = embed_pos.to(inputs_embeds.device)

#         hidden_states = inputs_embeds + embed_pos
#         hidden_states = self.layernorm_embedding(hidden_states)
#         hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

#         # expand attention_mask
#         if attention_mask is not None:
#             if self._use_flash_attention_2:
#                 attention_mask = attention_mask if 0 in attention_mask else None
#             elif self._use_sdpa and head_mask is None and not output_attentions:
#                 # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
#                 # the manual implementation that requires a 4D causal mask in all cases.
#                 # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#                 attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
#             else:
#                 # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#                 attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

#         encoder_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None

#         # check if head_mask has a correct number of layers specified if desired
#         if head_mask is not None:
#             if head_mask.size()[0] != (len(self.layers)):
#                 raise ValueError(
#                     f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
#                     f" {head_mask.size()[0]}."
#                 )

#         for idx, encoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 encoder_states = encoder_states + (hidden_states,)
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             to_drop = False
#             if self.training:
#                 dropout_probability = torch.rand([])
#                 if dropout_probability < self.layerdrop:  # skip the layer
#                     to_drop = True

#             if to_drop:
#                 layer_outputs = (None, None)
#             else:
#                 if self.gradient_checkpointing and self.training:
#                     layer_outputs = self._gradient_checkpointing_func(
#                         encoder_layer.__call__,
#                         hidden_states,
#                         attention_mask,
#                         (head_mask[idx] if head_mask is not None else None),
#                         output_attentions,
#                     )
#                 else:
#                     layer_outputs = encoder_layer(
#                         hidden_states,
#                         attention_mask,
#                         layer_head_mask=(head_mask[idx] if head_mask is not None else None),
#                         output_attentions=output_attentions,
#                     )

#                 hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#         return {'inputs_embeds': hidden_states, 'attention_mask':attention_mask}

# class BartEncoderTail(BartEncoderModelSplitter):
#     def __init__(self, config: BartConfig):
#         super().__init__(config)

#     def _clear_past_key_values(self):
#         self.past_key_values = None

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, BaseModelOutput]:
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # # retrieve input_ids and inputs_embeds
#         # if input_ids is not None and inputs_embeds is not None:
#         #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         # elif input_ids is not None:
#         #     input = input_ids
#         #     input_ids = input_ids.view(-1, input_ids.shape[-1])
#         # elif inputs_embeds is not None:
#         #     input = inputs_embeds[:, :, -1]
#         # else:
#         #     raise ValueError("You have to specify either input_ids or inputs_embeds")

#         # if inputs_embeds is None:
#         #     inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

#         # embed_pos = self.embed_positions(input)
#         # embed_pos = embed_pos.to(inputs_embeds.device)

#         # hidden_states = inputs_embeds + embed_pos
#         # hidden_states = self.layernorm_embedding(hidden_states)
#         # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

#         # VF-LLM: receive hidden_states from inputs_embeds directly
#         hidden_states = inputs_embeds
        
#         # # expand attention_mask
#         # if attention_mask is not None:
#         #     if self._use_flash_attention_2:
#         #         attention_mask = attention_mask if 0 in attention_mask else None
#         #     elif self._use_sdpa and head_mask is None and not output_attentions:
#         #         # output_attentions=True & head_mask can not be supported when using SDPA, fall back to
#         #         # the manual implementation that requires a 4D causal mask in all cases.
#         #         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#         #         attention_mask = _prepare_4d_attention_mask_for_sdpa(attention_mask, inputs_embeds.dtype)
#         #     else:
#         #         # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
#         #         attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

#         encoder_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None

#         # check if head_mask has a correct number of layers specified if desired
#         if head_mask is not None:
#             if head_mask.size()[0] != (len(self.layers)):
#                 raise ValueError(
#                     f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
#                     f" {head_mask.size()[0]}."
#                 )

#         for idx, encoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 encoder_states = encoder_states + (hidden_states,)
#             # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
#             to_drop = False
#             if self.training:
#                 dropout_probability = torch.rand([])
#                 if dropout_probability < self.layerdrop:  # skip the layer
#                     to_drop = True

#             if to_drop:
#                 layer_outputs = (None, None)
#             else:
#                 if self.gradient_checkpointing and self.training:
#                     layer_outputs = self._gradient_checkpointing_func(
#                         encoder_layer.__call__,
#                         hidden_states,
#                         attention_mask,
#                         (head_mask[idx] if head_mask is not None else None),
#                         output_attentions,
#                     )
#                 else:
#                     layer_outputs = encoder_layer(
#                         hidden_states,
#                         attention_mask,
#                         layer_head_mask=(head_mask[idx] if head_mask is not None else None),
#                         output_attentions=output_attentions,
#                     )

#                 hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#         if output_hidden_states:
#             encoder_states = encoder_states + (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
#         )




# class BartModelSplitter(BartModel, VFLModel):
#     def vfl_split(self, idx_of_layers: Iterable[int], is_encoder) -> bool:
#         return self._split_layers(idx_of_layers, is_encoder)

#     def _clear_past_key_values(self):
#         self.past_key_values = None
   
   
# class BartModelHead(BartModelSplitter,T5PreTrainedModel):
#     def __init__(self, config: T5Config):
#         super().__init__(config)
#         self.model_dim = config.d_model

#         self.shared = nn.Embedding(config.vocab_size, config.d_model)

#         encoder_config = copy.deepcopy(config)
#         encoder_config.is_decoder = False
#         encoder_config.use_cache = False
#         encoder_config.is_encoder_decoder = False
#         self.encoder = T5StackHead(encoder_config, self.shared)

#         ######### VF-LLM : decoder not needed in model head
#         # decoder_config = copy.deepcopy(config)
#         # decoder_config.is_decoder = True
#         # decoder_config.is_encoder_decoder = False
#         # decoder_config.num_layers = config.num_decoder_layers
#         # self.decoder = T5Stack(decoder_config, self.shared)

#         # self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#     def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
#         # print(f'T5ForConditionalGenerationHead _split_layers {list(idx_of_layers)}')
#         result = self.encoder.vfl_split(idx_of_layers)
#         self.config.num_layers = self.encoder.config.num_layers
#         self.config.num_decoder_layers = self.encoder.config.num_decoder_layers
#         return result

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         decoder_head_mask: Optional[torch.FloatTensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs
#     ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
#         # print('==== T5ForConditionalGenerationHead')
#         use_cache = False
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
#         if head_mask is not None and decoder_head_mask is None:
#             if self.config.num_layers == self.config.num_decoder_layers:
#                 warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
#                 decoder_head_mask = head_mask
        
#         # Encode if needed (training, first prediction pass)
#         encoder_intermediate = {}
#         if encoder_outputs is None:
#             # Convert encoder inputs in embeddings if needed
#             encoder_intermediate = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 inputs_embeds=inputs_embeds,
#                 head_mask=head_mask,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#             )
#             ### VF-LLM: add decode_input_ids
#             encoder_intermediate['decoder_input_ids']= decoder_input_ids
#             encoder_intermediate['labels']= labels
        
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )
#             encoder_intermediate = {'encoder_outputs':encoder_outputs,
#                     'decoder_input_ids':decoder_input_ids,
#                     'input_ids':input_ids}
        
#         return encoder_intermediate
        

# class T5ForConditionalGenerationBody(T5ForConditionalGenerationSplitter,T5PreTrainedModel):
#     def __init__(self, config: T5Config):
#         super().__init__(config)
#         self.model_dim = config.d_model

#         self.shared = nn.Embedding(config.vocab_size, config.d_model)

#         encoder_config = copy.deepcopy(config)
#         encoder_config.is_decoder = False
#         encoder_config.use_cache = False
#         encoder_config.is_encoder_decoder = False
#         self.encoder = T5StackTail(encoder_config, self.shared)

#         decoder_config = copy.deepcopy(config)
#         decoder_config.is_decoder = True
#         decoder_config.is_encoder_decoder = False
#         decoder_config.num_layers = config.num_decoder_layers
#         self.decoder = T5StackHead(decoder_config, self.shared)

#         # self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#     def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
#         # print(f'T5ForConditionalGenerationBody _split_layers {list(idx_of_layers)}')
#         origin_encoder_num = self.config.num_layers
#         origin_decoder_num = self.config.num_decoder_layers
#         self.encoder.vfl_split(idx_of_layers, is_encoder = 1)
#         self.config.num_layers = self.encoder.config.num_layers # encoder layers
        
#         decoder_idx_of_layers = []
#         for _id in idx_of_layers:
#             if (_id-origin_encoder_num) >= 0 :
#                 decoder_idx_of_layers.append( _id-origin_encoder_num )
#         self.decoder.vfl_split(decoder_idx_of_layers, is_encoder = 0)
#         self.config.num_decoder_layers = self.decoder.config.num_decoder_layers # decoder layers
#         return True

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         decoder_head_mask: Optional[torch.FloatTensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         encoder_forward = False,
#         **kwargs
#     ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
#         # print('==== T5ForConditionalGenerationBody')
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         use_cache = False

#         # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
#         if head_mask is not None and decoder_head_mask is None:
#             if self.config.num_layers == self.config.num_decoder_layers:
#                 warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
#                 decoder_head_mask = head_mask

#         # Encode if needed (training, first prediction pass)
#         if encoder_outputs is None:
#             # Convert encoder inputs in embeddings if needed
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 inputs_embeds=inputs_embeds,
#                 head_mask=head_mask,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 **kwargs
#             )
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )

#         ############## VF-LLM: only do encoder forward
#         if encoder_forward:
#             return encoder_outputs

#         hidden_states = encoder_outputs[0]

#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)

#         if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
#             # get decoder inputs from shifting lm labels to the right
#             decoder_input_ids = self._shift_right(labels)

#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)
#             hidden_states = hidden_states.to(self.decoder.first_device)
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(self.decoder.first_device)
#             if decoder_attention_mask is not None:
#                 decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

#         # Decode
#         decoder_intermediate = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             inputs_embeds=decoder_inputs_embeds,
#             past_key_values=past_key_values,
#             encoder_hidden_states=hidden_states,
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         decoder_intermediate['encoder_outputs'] = encoder_outputs
#         return decoder_intermediate

# class T5ForConditionalGenerationTail_3slice(T5ForConditionalGenerationSplitter,T5PreTrainedModel):
#     def __init__(self, config: T5Config):
#         super().__init__(config)
#         self.model_dim = config.d_model

#         self.shared = nn.Embedding(config.vocab_size, config.d_model)

#         ####### VF-LLM: no encoder needed here
#         # encoder_config = copy.deepcopy(config)
#         # encoder_config.is_decoder = False
#         # encoder_config.use_cache = False
#         # encoder_config.is_encoder_decoder = False
#         # self.encoder = T5Stack(encoder_config, self.shared)

#         decoder_config = copy.deepcopy(config)
#         decoder_config.is_decoder = True
#         decoder_config.is_encoder_decoder = False
#         decoder_config.num_layers = config.num_decoder_layers
#         self.decoder = T5StackTail(decoder_config, self.shared)

#         self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#     def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
#         # print(f'T5ForConditionalGenerationTail _split_layers {list(idx_of_layers)}')
#         origin_encoder_num = self.config.num_layers
#         origin_decoder_num = self.config.num_decoder_layers

#         decoder_idx_of_layers = []
#         for _id in idx_of_layers:
#             if (_id-origin_encoder_num) >= 0 :
#                 decoder_idx_of_layers.append( _id-origin_encoder_num )
#         result = self.decoder.vfl_split(decoder_idx_of_layers, is_encoder = 0)

#         self.config.num_layers = self.decoder.config.num_layers
#         self.config.num_decoder_layers = self.decoder.config.num_decoder_layers
#         return result

#     @property
#     def head_layer(self):
#         return self.lm_head

#     @head_layer.setter
#     def head_layer(self, lm_head):
#         self.lm_head = lm_head


#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         decoder_head_mask: Optional[torch.FloatTensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs
#     ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
#         # print('==== T5ForConditionalGenerationTail_3slice')
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         use_cache = False

#         # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
#         if head_mask is not None and decoder_head_mask is None:
#             if self.config.num_layers == self.config.num_decoder_layers:
#                 warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
#                 decoder_head_mask = head_mask

#         ############# VF-LLM: encode done in previous model slices ######
#         # # Encode if needed (training, first prediction pass)
#         # if encoder_outputs is None:
#         #     # Convert encoder inputs in embeddings if needed
#         #     encoder_outputs = self.encoder(
#         #         input_ids=input_ids,
#         #         attention_mask=attention_mask,
#         #         inputs_embeds=inputs_embeds,
#         #         head_mask=head_mask,
#         #         output_attentions=output_attentions,
#         #         output_hidden_states=output_hidden_states,
#         #         return_dict=return_dict,
#         #         **kwargs
#         #     )
#         # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#         #     encoder_outputs = BaseModelOutput(
#         #         last_hidden_state=encoder_outputs[0],
#         #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#         #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#         #     )

#         # hidden_states = encoder_outputs[0] 

#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)

#         if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
#             # get decoder inputs from shifting lm labels to the right
#             decoder_input_ids = self._shift_right(labels)

#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)
#             hidden_states = hidden_states.to(self.decoder.first_device)
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(self.decoder.first_device)
#             if decoder_attention_mask is not None:
#                 decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

#         # Decode
#         decoder_inputs_embeds = inputs_embeds # receive decoder intermediates from model body

#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=attention_mask, #attention_mask=decoder_attention_mask,
#             inputs_embeds=decoder_inputs_embeds,
#             past_key_values=past_key_values,
#             # encoder_hidden_states=hidden_states, # receive encoder hidden from model body -- in kwargs
#             # encoder_attention_mask=attention_mask, # receive encoder hidden from model body -- in kwargs
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             **kwargs
#         )
        
#         sequence_output = decoder_outputs[0]

#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.encoder.first_device)
#             self.lm_head = self.lm_head.to(self.encoder.first_device)
#             sequence_output = sequence_output.to(self.lm_head.weight.device)

#         if self.config.tie_word_embeddings:
#             # Rescale output before projecting on vocab
#             # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
#             sequence_output = sequence_output * (self.model_dim**-0.5)

#         lm_logits = self.lm_head(sequence_output)
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-100)
#             # move labels to correct device to enable PP
#             labels = labels.to(lm_logits.device)
#             loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
#             # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

#         if not return_dict:
#             output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
#             return ((loss,) + output) if loss is not None else output

#         return Seq2SeqLMOutput(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )
    
#     ## Rewrite for MainTaskVFL_LLM to inherit (generation related utils)
#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def get_output_embeddings(self):
#         return self.lm_head

#     def get_encoder(self):
#         return self.parties[0].local_model.encoder
#         #,self.parties[1].global_model.encoder 
#         #self.encoder

#     def get_decoder(self):
#         return self.parties[1].global_model.decoder #self.decoder
    
#     def _prepare_encoder_decoder_kwargs_for_generation(
#         self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
#     ):
#         # 1. get encoder
#         # encoder = self.get_encoder()
#         head_encoder = self.get_encoder()

#         # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
#         # as the inputs.
#         if hasattr(self, "hf_device_map"):
#             if hasattr(encoder, "_hf_hook"):
#                 encoder._hf_hook.io_same_device = True
#             else:
#                 add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

#         # 2. Prepare encoder args and encoder kwargs from model kwargs.
#         irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
#         encoder_kwargs = {
#             argument: value
#             for argument, value in model_kwargs.items()
#             if not any(argument.startswith(p) for p in irrelevant_prefix)
#         }
#         encoder_signature = set(inspect.signature(head_encoder.forward).parameters) #set(inspect.signature(encoder.forward).parameters)
#         encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
#         if not encoder_accepts_wildcard:
#             encoder_kwargs = {
#                 argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
#             }

#         # 3. make sure that encoder returns `ModelOutput`
#         model_input_name = model_input_name if model_input_name is not None else self.main_input_name
#         encoder_kwargs["return_dict"] = True
#         encoder_kwargs[model_input_name] = inputs_tensor

#         head_encoder_output = self.parties[0].local_model(**encoder_kwargs)
#         model_kwargs["encoder_outputs"]: ModelOutput = self.parties[1].global_model( **head_encoder_output, encoder_forward = 1)
#         # encoder(**encoder_kwargs)

#         return model_kwargs



# class T5ForConditionalGenerationTail_2slice(T5ForConditionalGenerationSplitter,T5PreTrainedModel):
#     def __init__(self, config: T5Config):
#         super().__init__(config)
#         self.model_dim = config.d_model

#         self.shared = nn.Embedding(config.vocab_size, config.d_model)

#         encoder_config = copy.deepcopy(config)
#         encoder_config.is_decoder = False
#         encoder_config.use_cache = False
#         encoder_config.is_encoder_decoder = False
#         self.encoder = T5StackTail(encoder_config, self.shared)

#         decoder_config = copy.deepcopy(config)
#         decoder_config.is_decoder = True
#         decoder_config.is_encoder_decoder = False
#         decoder_config.num_layers = config.num_decoder_layers
#         self.decoder = T5Stack(decoder_config, self.shared)

#         self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#     @property
#     def head_layer(self):
#         return self.lm_head

#     @head_layer.setter
#     def head_layer(self, lm_head):
#         self.lm_head = lm_head

#     def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
#         # print(f'T5ForConditionalGenerationTail _split_layers {list(idx_of_layers)}')
#         result = self.encoder.vfl_split(idx_of_layers)
#         self.config.num_layers = self.encoder.config.num_layers
#         self.config.num_decoder_layers = self.encoder.config.num_decoder_layers
#         return result

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         decoder_input_ids: Optional[torch.LongTensor] = None,
#         decoder_attention_mask: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         decoder_head_mask: Optional[torch.FloatTensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs
#     ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
#         if head_mask is not None and decoder_head_mask is None:
#             if self.config.num_layers == self.config.num_decoder_layers:
#                 warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
#                 decoder_head_mask = head_mask

#         # Encode if needed (training, first prediction pass)
#         if encoder_outputs is None:
#             # Convert encoder inputs in embeddings if needed
#             encoder_outputs = self.encoder(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 inputs_embeds=inputs_embeds,
#                 head_mask=head_mask,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict,
#                 **kwargs
#             )
#         elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
#             encoder_outputs = BaseModelOutput(
#                 last_hidden_state=encoder_outputs[0],
#                 hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
#                 attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
#             )

#         hidden_states = encoder_outputs[0]

#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)

#         if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
#             # get decoder inputs from shifting lm labels to the right
#             decoder_input_ids = self._shift_right(labels)

#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.decoder.first_device)
#             hidden_states = hidden_states.to(self.decoder.first_device)
#             if decoder_input_ids is not None:
#                 decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
#             if attention_mask is not None:
#                 attention_mask = attention_mask.to(self.decoder.first_device)
#             if decoder_attention_mask is not None:
#                 decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

#         # Decode
#         decoder_outputs = self.decoder(
#             input_ids=decoder_input_ids,
#             attention_mask=decoder_attention_mask,
#             inputs_embeds=decoder_inputs_embeds,
#             past_key_values=past_key_values,
#             encoder_hidden_states=hidden_states,
#             encoder_attention_mask=attention_mask,
#             head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = decoder_outputs[0]

#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.encoder.first_device)
#             self.lm_head = self.lm_head.to(self.encoder.first_device)
#             sequence_output = sequence_output.to(self.lm_head.weight.device)

#         if self.config.tie_word_embeddings:
#             # Rescale output before projecting on vocab
#             # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
#             sequence_output = sequence_output * (self.model_dim**-0.5)

#         lm_logits = self.lm_head(sequence_output)
#         loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss(ignore_index=-100)
#             # move labels to correct device to enable PP
#             labels = labels.to(lm_logits.device)
#             loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
#             # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

#         if not return_dict:
#             output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
#             return ((loss,) + output) if loss is not None else output

#         return Seq2SeqLMOutput(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=decoder_outputs.past_key_values,
#             decoder_hidden_states=decoder_outputs.hidden_states,
#             decoder_attentions=decoder_outputs.attentions,
#             cross_attentions=decoder_outputs.cross_attentions,
#             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
#             encoder_hidden_states=encoder_outputs.hidden_states,
#             encoder_attentions=encoder_outputs.attentions,
#         )



# class ModelPartitionPipelineT5(ModelPartitionPipeline):

#     def _load_model_head(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
#         model_head = T5ForConditionalGenerationHead.from_pretrained(model_name_or_path, **kwargs)
#         if do_split:
#             self.all_layer_num = model_head.config.num_hidden_layers
#             split_range = range(0, self.split_index[0])
#             model_head.vfl_split(split_range)

#         return model_head#.to(self.device)

#     def _load_model_tail(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
#         if self.args.model_architect == 'CLM':
#             if self.args.vfl_model_slice_num == 2:
#                 model_tail = T5ForConditionalGenerationTail_2slice.from_pretrained(model_name_or_path, **kwargs)
#             else:
#                 model_tail = T5ForConditionalGenerationTail_3slice.from_pretrained(model_name_or_path, **kwargs)
#         else:
#             raise ValueError(f"model_architect {self.args.model_architect} not supported for {model_name_or_path}")
        
#         if do_split:
#             if self.num_of_slices == 2:
#                 split_range = range(self.split_index[0],model_tail.config.num_layers)
#             else:
#                 split_range = range(model_tail.config.num_layers+model_tail.config.num_decoder_layers-self.split_index[1],model_tail.config.num_layers+model_tail.config.num_decoder_layers)
            
#             model_tail.vfl_split(split_range)
#             # print(list(split_range))
#             # print(f'Model Tail:{len(model_tail.model.h)} {do_split}')


#         return model_tail#.to(self.device)
    
#     def _load_model_body(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
#         model_body = T5ForConditionalGenerationBody.from_pretrained(model_name_or_path, **kwargs)
#         if do_split:
#             split_range = range(self.split_index[0], model_body.config.num_layers+model_body.config.num_decoder_layers-self.split_index[1])
            
#             model_body.vfl_split(split_range)
            
#             # print(list(split_range))
#             # print(f'Model Body:{len(model_body.h)} {do_split}')
           
        
#         return model_body#.to(self.device)
