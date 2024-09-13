"""
copy source codes from transformers, then modify
code based on transformers=4.37.2
"""
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.bert.modeling_bert import *

from torch.nn import ModuleList, Parameter
from typing import Iterable, Optional, Union, List, Tuple, Callable, Dict, Iterator
from loguru import logger
import torch
import copy
import os
from peft.peft_model import PeftModel
from .base import ModelPartitionPipeline, VFLModel

class BertModelSplitter(BertModel, VFLModel):
    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self._split_layers(idx_of_layers)

    def _split_layers(self, idx_of_layers: Iterable[int]) -> bool:
        new_layers = ModuleList()
        for i, layer in enumerate(self.encoder.layer):
            if i in idx_of_layers:
                new_layers.append(layer)
        self.encoder.layer = new_layers

        # update config
        self.config.num_hidden_layers = len(new_layers)        
        return True

    def _clear_past_key_values(self):
        self.past_key_values = None
    
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


class BertModelHead(BertModelSplitter):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.past_key_values = None
        self.embedding_output = None
        
        del self.pooler
        # defense related
        self.inner_mid_model = None

    def _clear_past_key_values(self):
        self.past_key_values = None

    def get_input_embeddings(self):
        if self.embeddings != None:
            return self.embeddings.word_embeddings
        else:
            return None

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # print('input_shape:', input_shape)

        batch_size, seq_length = input_shape[:2]
        # print('batch_size:',batch_size,' seq_length',seq_length)

        batch_size = int(batch_size)
        seq_length = int(seq_length)

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # print('bs:',batch_size,'  seq:',seq_length)  # [2048,1029]
        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        # print('attention_mask:',attention_mask.shape,attention_mask)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                # print('self.embeddings.token_type_ids:',self.embeddings.token_type_ids.shape)
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                # print('buffered_token_type_ids:',type(buffered_token_type_ids),buffered_token_type_ids.shape)
                # [1,512]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                # print('buffered_token_type_ids_expanded:',type(buffered_token_type_ids_expanded),buffered_token_type_ids_expanded.shape)

                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # print('token_type_ids:',token_type_ids.shape,token_type_ids)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        self.embedding_output = embedding_output

        ############ mid ############
        if self.inner_mid_model != None:
            # print(' =======  Inner MID  ======= ')
            embedding_output, self.mid_loss = self.inner_mid_model(embedding_output)
            self.embedding_output = embedding_output
        ############ mid ############

        encoder_intermediate = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return {'inputs_embeds': encoder_intermediate.last_hidden_state,
                'attention_mask': attention_mask}


class BertModelBody(BertModelSplitter):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.past_key_values = None

        del self.pooler
        # del self.embeddings
        # self.embeddings = None

        # todo: del norm will cause error when load from original model weight
        # del self.norm
        
    def get_input_embeddings(self):
        if self.embeddings == None:
            return None
        else: 
            return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            inputs_embeds: Optional[torch.Tensor] = None,  # local_pred: intermediate_embedding
            attention_mask: Optional[torch.Tensor] = None,  # local_attention_mask
            input_ids: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # batch_size, seq_length = input_shape[:2]
        # device = intermediate[0].device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        # print('attention_mask:',attention_mask.shape,attention_mask)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                # print('buffered_token_type_ids:',type(buffered_token_type_ids),buffered_token_type_ids.shape)
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                # print('buffered_token_type_ids_expanded:',type(buffered_token_type_ids_expanded),buffered_token_type_ids_expanded.shape)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        # print('token_type_ids:',token_type_ids.shape,token_type_ids)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_intermediate = self.encoder(
            hidden_states=inputs_embeds,  # intermediate,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return {'inputs_embeds': encoder_intermediate.last_hidden_state,
                'attention_mask': attention_mask}


class BertModelTail(BertModelSplitter):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.past_key_values = None
        
        # todo: del norm will cause error when load from original model weight
        # del self.embeddings
        # self.embeddings = None

    def get_input_embeddings(self):
        if self.embeddings == None:
            return None
        else: 
            return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # batch_size, seq_length = input_shape[:2]
        # device = intermediate[0].device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        # print('attention_mask:',attention_mask.shape,attention_mask)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                # print('buffered_token_type_ids:',type(buffered_token_type_ids),buffered_token_type_ids.shape)
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                # print('buffered_token_type_ids_expanded:',type(buffered_token_type_ids_expanded),buffered_token_type_ids_expanded.shape)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        # print('token_type_ids:',token_type_ids.shape,token_type_ids)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # print(f'Tail inputs_embeds:{inputs_embeds[0,0,:5]}')
        # print(f'Tail extended_attention_mask:{extended_attention_mask}')
        # print(f'Tail head_mask:{head_mask}')
        # print(f'Tail encoder_hidden_states:{encoder_hidden_states}')
        # if past_key_values!= None:
        #     print(f'Tail past_key_values:{len(past_key_values)} {past_key_values[0]}')
        encoder_outputs = self.encoder(
            hidden_states=inputs_embeds,  # intermediate,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # print('Tail sequence_output:',sequence_output[0,:5],' pooled_output:',pooled_output[0,:5])
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

# Global Model Wrapper
class BertTailForCausalLM(BertLMHeadModel, VFLModel):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(config)
        self.bert = BertModelTail(config)
        # Initialize weights and apply final processing
        self.post_init()

    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.bert.vfl_split(idx_of_layers)

    def _clear_past_key_values(self):
        self.bert._clear_past_key_values()

    @property
    def head_layer(self):
        return self.cls

    @head_layer.setter
    def head_layer(self, cls):
        self.cls = cls

class BertTailForQuestionAnswering(BertForQuestionAnswering, VFLModel):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(config)
        self.bert = BertModelTail(config)
        # Initialize weights and apply final processing
        self.post_init()

    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.bert.vfl_split(idx_of_layers)

    def _clear_past_key_values(self):
        self.bert._clear_past_key_values()

    @property
    def head_layer(self):
        return self.qa_outputs

    @head_layer.setter
    def head_layer(self, qa_outputs):
        self.qa_outputs = qa_outputs

class BertTailForSequenceClassification(BertForSequenceClassification, VFLModel):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(config)
        self.bert = BertModelTail(config)
        # Initialize weights and apply final processing
        self.post_init()

    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self.bert.vfl_split(idx_of_layers)

    def _clear_past_key_values(self):
        self.bert._clear_past_key_values()

    @property
    def head_layer(self):
        return self.classifier

    @head_layer.setter
    def head_layer(self, classifier):
        self.classifier = classifier

class ModelPartitionPipelineBert(ModelPartitionPipeline):

    def _load_model_head(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_head = BertModelHead.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            self.all_layer_num = model_head.config.num_hidden_layers
            split_range = range(0, self.split_index[0])
            model_head.vfl_split(split_range)
            # print(list(split_range))
            # print(f'Model Head:{len(model_head.h)} {do_split}')

        return model_head.to(self.device)

    def _load_model_tail(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        if self.args.model_architect == 'CLM':
            model_tail = BertTailForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        elif self.args.model_architect == 'CLS':
            model_tail = BertTailForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        elif self.args.model_architect == 'TQA':
            model_tail = BertTailForQuestionAnswering.from_pretrained(model_name_or_path, **kwargs)
        else:
            raise ValueError(f"model_architect {self.args.model_architect} not supported for {model_name_or_path}")
        
        if do_split:
            if self.num_of_slices == 2:
                split_range = range(self.split_index[0],model_tail.config.num_hidden_layers)
            else:
                split_range = range(model_tail.config.num_hidden_layers-self.split_index[1],model_tail.config.num_hidden_layers)
            model_tail.vfl_split(split_range)
            # print(list(split_range))
            # print(f'Model Tail:{len(model_tail.transformer.h)} {do_split}')

        return model_tail.to(self.device)#.to(self.device)


    def _load_model_body(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_body = BertModelBody.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            split_range = range(self.split_index[0], model_body.config.num_hidden_layers-self.split_index[1])
            model_body.vfl_split(split_range)
            
            # print(list(split_range))
            # print(f'Model Body:{len(model_body.h)} {do_split}')
           
        
        return model_body.to(self.device)
