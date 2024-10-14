"""
copy source codes from transformers, then modify
code based on transformers=4.37.2
"""
from transformers.modeling_outputs import CausalLMOutputWithPast

from torch.nn import ModuleList, Parameter
from typing import Iterable, Optional, Union, List, Tuple, Callable, Dict, Iterator
from loguru import logger
import torch
import copy
import os
from peft.peft_model import PeftModel
from .base import ModelPartitionPipeline, VFLModel
from PIL import Image


from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from .third_party_modeling.configuration_minicpm import MiniCPMConfig
from .third_party_modeling.modeling_minicpm import *
from .third_party_modeling.modeling_minicpmv import *

from .minicpm import *
# from .third_party_modeling.tokenization_minicpm import ChatGLMTokenizer

class MiniCPMVModelSplitter(MiniCPMV, VFLModel):
    def vfl_split(self, idx_of_layers: Iterable[int]) -> bool:
        return self._split_layers(idx_of_layers)

    def _split_layers(self, idx_of_layers: Iterable[int]) -> bool:
        new_layers = ModuleList()
        for i, layer in enumerate(self.llm.model.layers):
            if i in idx_of_layers:
                new_layers.append(layer)
        self.llm.model.layers = new_layers

        # update config
        self.llm.model.config.num_hidden_layers = len(new_layers)
        self.llm.config.num_hidden_layers = len(new_layers)
        self.config.num_hidden_layers = len(new_layers)
        return True

    def _clear_past_key_values(self):
        self.past_key_values = None
    

    
class MiniCPMVModelHead(MiniCPMVModelSplitter, MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super(MiniCPMVPreTrainedModel,self).__init__(config)

        self.llm = MiniCPMHeadForCausalLM(config) #MiniCPMForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.transform = self.init_transform()

    
    def _clear_past_key_values(self):
        self.llm._clear_past_key_values()

    def forward(self, samples=None, question=None, image=None, **kwargs):
        if samples == None:
            if not (question == None and image==None):
                samples = {'question':question, 'image':image}

        if samples != None:
            print('sample forward')
            vllm_embedding, vision_hidden_states = self.get_vllm_embedding(samples)
            
            position_ids = samples["position_ids"]
            if position_ids.dtype != torch.int64:
                position_ids = position_ids.long()
            
            return self.llm(
                input_ids=None,
                position_ids=position_ids,
                inputs_embeds=vllm_embedding,
                **kwargs
            )
        else:
            print('direct forward model head')
            return self.llm(
                **kwargs
            )
    
    def get_vision_embedding(self, pixel_values):
        res = []

        try:
            dtype = self.llm.lm_head.weight.dtype # for MiniCPMForCausalLM
        except:
            dtype = self.llm.model.layers[0].mlp.gate_proj.weight.dtype # for MiniCPMModelHead
        
        def process_each_pixel(pixel_value, dtype, config, vpm, resampler):
            H, W = pixel_value.shape[-2:]
            target_size = (math.ceil(H / config.patch_size), math.ceil(W / config.patch_size))
            vision_embedding = self.vpm_forward_features(pixel_value.unsqueeze(0).type(dtype))
            
            if hasattr(vpm, 'num_prefix_tokens') and vpm.num_prefix_tokens > 0:
                vision_embedding = vision_embedding[:, vpm.num_prefix_tokens:]
            return resampler(vision_embedding, target_size)

        for pixel_value in pixel_values:
            result = process_each_pixel(pixel_value, dtype, self.config, self.vpm, self.resampler)
            res.append(result)
        return torch.vstack(res)

    def get_vllm_embedding(self, data):
        # print('pixel_values:',len(data['pixel_values']) )
        # print('image_bound:',len(data['image_bound']),data['image_bound'][0].shape )

        if "vision_hidden_states" not in data:
            pixel_values_list = data["pixel_values"]
            # pixel_values_list: list of bs [[tensor 3, 1024, 973], [tensor],.. ] 3, 1024, 973

            vision_hidden_states = []
            for pixel_values in pixel_values_list:
                if len(pixel_values) > 0:
                    vision_hidden_states.append(self.get_vision_embedding(pixel_values))
                elif self.training:
                    dtype = self.llm.lm_head.weight.dtype
                    device = self.llm.lm_head.weight.device
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224), device=device, dtype=dtype
                    )
                    vision_hidden_states.append(self.get_vision_embedding(dummy_image))
                else:
                    vision_hidden_states.append([])
        else:
            vision_hidden_states = data["vision_hidden_states"]
        # vision_hidden_states: list of 4 [1,64,2304]

        # add to receive multiple input types
        if 'vllm_embedding' in data.keys() and data['vllm_embedding'] != None:
            vllm_embedding = (
                data['vllm_embedding'] * self.llm.config.scale_emb
            )# 4, 160, 2304 
            print('-model head vllm_embedding:',vllm_embedding[0,:5])

        else:
            vllm_embedding = (
                self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
                # self.llm.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
            )# 4, 160, 2304

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        
        # bs = len(data["input_ids"])
        # add to receive multiple input types
        if 'input_ids' in data.keys():
            bs = len(data["input_ids"])
        else:
            bs = len(data["pixel_values"])

        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i] # tensor 5 64 2304
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i] # 358, 2304

                cur_image_bound = data["image_bound"][i] # 5, 2
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [
                            torch.arange(r[0], r[1], dtype=torch.long)
                            for r in cur_image_bound
                        ]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0
        return vllm_embedding, vision_hidden_states
    
    # inherit&alter generation reltaed methods from MiniCPMForCausalLM into VFL form
    def pre_generation(self,
            samples = None, # added to accord withh MainTaskVFL_LLM.mm_generate()
            data_list=None,
            img_list=None,
            tokenizer=None,
            max_inp_length: Optional[int] = None,
            vision_hidden_states=None,
            return_vision_hidden_states=False,
            **kwargs
        ):
        if samples != None:
            data_list = samples['question']
            img_list = [[_imgs] for _imgs in samples['image']]

        assert data_list is not None
        bs = len(data_list)
        if img_list == None:
            img_list = [[] for i in range(bs)]
        assert bs == len(img_list)

        model_inputs = self._process_list(tokenizer, data_list, max_inp_length)

        if vision_hidden_states is None:
            pixel_values = []
            for i in range(bs):
                img_inps = []
                for img in img_list[i]:
                    img_inps.append(self.transform(img).to(self.device))
                if img_inps:
                    pixel_values.append(img_inps)
                else:
                    pixel_values.append([])
            model_inputs["pixel_values"] = pixel_values
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = self.get_vllm_embedding(model_inputs)
        
        
        aligned_input = {'inputs_embeds':model_inputs["inputs_embeds"].to(self.device)} #,'kwargs':kwargs}
        aligned_input.update(kwargs)
        aligned_input['tokenizer'] = tokenizer
        aligned_input['max_inp_length'] = max_inp_length
        aligned_input['vision_hidden_states'] = vision_hidden_states
        aligned_input['return_vision_hidden_states'] = return_vision_hidden_states

        return aligned_input
    
    def pre_chat(self, sample, context, # image, msgs,
        tokenizer, vision_hidden_states=None, max_new_tokens=1024,
        sampling=True, max_inp_length=2048,
        **kwargs):
        image = sample['image'][0] # type = Image
        msgs = [{'role': 'user', 'content': sample['question'][0]}] 

        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        # msgs to prompt
        prompt = ""
        for i, msg in enumerate(msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
                if self.config.slice_mode:
                    images, final_placeholder = self.get_slice_image_placeholder(
                        image, tokenizer
                    )
                    content = final_placeholder + "\n" + content
                else:
                    images = [image]
                    content = (
                        tokenizer.im_start
                        + tokenizer.unk_token * self.config.query_num
                        + tokenizer.im_end
                        + "\n"
                        + content
                    )
            prompt += "<用户>" if role == "user" else "<AI>"
            prompt += content
        prompt += "<AI>"
        final_input = prompt

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        aligned_input = {
            "data_list":[final_input], "max_inp_length":max_inp_length, 
            "img_list":[images],"tokenizer":tokenizer,
            "max_new_tokens":max_new_tokens,
            "vision_hidden_states":vision_hidden_states,
            "return_vision_hidden_states":True,
            # "msgs":msgs
            # "generation_config":generation_config,
            }
        return aligned_input, generation_config

class MiniCPMVModelBody(MiniCPMVModelSplitter, MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super(MiniCPMVPreTrainedModel,self).__init__(config)

        self.llm = MiniCPMTailForCausalLM(config) #MiniCPMForCausalLM(config)
        # self.vpm = self.init_vision_module()
        # self.vision_dim = self.vpm.embed_dim
        # self.embed_dim = self.llm.config.hidden_size
        # self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        # self.transform = self.init_transform()
    
    def _clear_past_key_values(self):
        self.llm._clear_past_key_values()

    def forward(self, **kwargs):
        # vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        # position_ids = data["position_ids"]
        # if position_ids.dtype != torch.int64:
        #     position_ids = position_ids.long()

        return self.llm(
            **kwargs
        )

class MiniCPMVModelTail(MiniCPMVModelSplitter,MiniCPMForCausalLM, MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super(MiniCPMVPreTrainedModel,self).__init__(config)
        self.llm = MiniCPMTailForCausalLM(config) #MiniCPMForCausalLM(config)
        # self.vpm = self.init_vision_module()
        # self.vision_dim = self.vpm.embed_dim
        # self.embed_dim = self.llm.config.hidden_size
        # self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        # self.transform = self.init_transform()
    
    def _clear_past_key_values(self):
        self.llm._clear_past_key_values()

    def forward(self, **kwargs):
        return self.llm(
            **kwargs
        )

    # inherit&alter MiniCPMV generation methods into VFL form --  inherited by MainTaskVFL_LLM
    def generate(
        self,
        data_list=None,
        img_list=None,
        tokenizer=None,
        max_inp_length: Optional[int] = None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        llm=None,
        **kwargs
    ):
        if data_list != None: # normal generation: do MiniCPMVModelHead.pre_generation(input align) first
            aligned_input = self.pre_generation(data_list=data_list,
                        img_list=img_list,
                        tokenizer=tokenizer,
                        max_inp_length = max_inp_length,
                        vision_hidden_states=vision_hidden_states,
                        return_vision_hidden_states=return_vision_hidden_states,
                        **kwargs)
            inputs_embeds = aligned_input['inputs_embeds']
            vision_hidden_states = aligned_input['vision_hidden_states']
            kwargs = aligned_input['kwargs']
        else: # pre_generation(input align) already done in model head
            inputs_embeds = kwargs.get('inputs_embeds')
            del kwargs['inputs_embeds']

        with torch.inference_mode():
            output = super(MiniCPMForCausalLM, self).generate(
                inputs_embeds=inputs_embeds.to(self.device),
                pad_token_id=0,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
            result = output
            
        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result
    
 
    def chat(
        self, final_input, images, generation_config, tokenizer, 
        vision_hidden_states=None,
        max_new_tokens=1024,
        sampling=True,
        max_inp_length=2048,
        **kwargs
    ):  

        # already done in model_head.pre_chat
        # final_input, images, generation_config = self.pre_chat(image, msgs, context, tokenizer,
        #     vision_hidden_states, max_new_tokens,
        #     sampling, max_inp_length,
        #     **kwargs)
        with torch.inference_mode():
            res, vision_hidden_states = self.generate(
                data_list=[final_input],
                max_inp_length=max_inp_length,
                img_list=[images],
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                return_vision_hidden_states=True,
                **generation_config
            )
        answer = res[0]
        context = msgs.copy()
        context.append({"role": "assistant", "content": answer})

        return answer, context, generation_config



# Global Model Wrapper

class ModelPartitionPipelineMiniCPMV(ModelPartitionPipeline):

    def _load_model_head(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_head = MiniCPMVModelHead.from_pretrained(model_name_or_path, **kwargs)

        if do_split:
            self.all_layer_num = model_head.config.num_hidden_layers
            split_range = range(0, self.split_index[0])
            model_head.vfl_split(split_range)

        return model_head.to(self.device)

    def _load_model_tail(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        if self.args.model_architect == 'MM':
            model_tail = MiniCPMVModelTail.from_pretrained(model_name_or_path, **kwargs)
        else:
            raise ValueError(f"model_architect {self.args.model_architect} not supported for {model_name_or_path}")
        
        if do_split:
            if self.num_of_slices == 2:
                split_range = range(self.split_index[0],model_tail.config.num_hidden_layers)
            else:
                split_range = range(model_tail.config.num_hidden_layers-self.split_index[1],model_tail.config.num_hidden_layers)
            model_tail.vfl_split(split_range)

        return model_tail.to(self.device)

    def _load_model_body(self, model_name_or_path, do_split=False, **kwargs) -> Union[PreTrainedModel, VFLModel]:
        model_body = MiniCPMVModelBody.from_pretrained(model_name_or_path, **kwargs)
        if do_split:
            split_range = range(self.split_index[0], model_body.config.num_hidden_layers-self.split_index[1])
            model_body.vfl_split(split_range)
           
        # if self.args.model_architect == 'MM':
        #     model_body = MiniGPT4Body(model_body, self.args.tokenizer)

        return model_body.to(self.device)
