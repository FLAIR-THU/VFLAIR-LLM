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

from models.llm_models.minigpt4.minigpt4 import MiniGPT4Head, MiniGPT4Body, MiniGPT4Tail
from transformers import StoppingCriteria, StoppingCriteriaList
from models.llm_models.minigpt4.conversation import StoppingCriteriaSub # minigpt4.conversation.conversation

from .third_party_modeling.configuration_minicpm import MiniCPMVConfig
from .third_party_modeling.modeling_minicpmv import *
# from .third_party_modeling.tokenization_minicpm import ChatGLMTokenizer

from .minicpm import *
class MiniCPMVModelHead(MiniCPMV):
    def __init__(self, llm):
        super().__init__(llm.config)

        self.llm = llm #MiniCPMForCausalLM(config) 
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.transform = self.init_transform()

    def forward(self, samples, **kwargs):
        print('model head samples:',samples.keys()) # question  image
        print('model head kwargs:',kwargs.keys()) # labels

        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(samples)
        position_ids = samples["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()
        
        print('vllm_embedding inputs_embeds:',vllm_embedding.shape)
        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )
    
    def get_vllm_embedding(self, data):
        if "vision_hidden_states" not in data:
            pixel_values_list = data["pixel_values"]
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

        vllm_embedding = (
            # self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
            self.llm.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        )
        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
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

    def pre_generation(self,
            data_list=None,
            img_list=None,
            tokenizer=None,
            max_inp_length: Optional[int] = None,
            vision_hidden_states=None,
            return_vision_hidden_states=False,
            **kwargs
        ):
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

        # print('pre_generation output inputs_embeds:',aligned_input["inputs_embeds"].shape)
        # print('pre_generation output aligned_input:',aligned_input.keys())

        return aligned_input
        
class MiniCPMVModelBody(MiniCPMV):
    def __init__(self, llm):
        super().__init__(llm.config)

        self.llm = llm #MiniCPMForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.transform = self.init_transform()

    def forward(self, **kwargs):
        # vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        # position_ids = data["position_ids"]
        # if position_ids.dtype != torch.int64:
        #     position_ids = position_ids.long()
        print('model body input:',kwargs.keys())
        return self.llm(
            **kwargs
        )

class MiniCPMVModelTail(MiniCPMV):
    def __init__(self, llm):
        super().__init__(llm.config)

        self.llm = llm #MiniCPMForCausalLM(config)
        print('MiniCPMVModelTail init llm:',type(llm))

        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.transform = self.init_transform()

    def forward(self, **kwargs):
        # vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        # position_ids = data["position_ids"]
        # if position_ids.dtype != torch.int64:
        #     position_ids = position_ids.long()
        print('model tail input:',kwargs.keys())
        return self.llm(
            **kwargs
        )

    def generate(
        self,
        data_list=None,
        img_list=None,
        tokenizer=None,
        max_inp_length: Optional[int] = None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        **kwargs
    ):
        print('--- MiniCPMVModelTail generate')
        if data_list != None:
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
        else:
            inputs_embeds = kwargs.get('inputs_embeds')
            del kwargs['inputs_embeds']

        with torch.inference_mode():
            print('llm.generate inputs_embeds:',inputs_embeds.shape)
            print('llm.generate kwargs:',kwargs.keys())
            print('self.llm:',type(self.llm))
            output = self.llm.generate(
                inputs_embeds=inputs_embeds.to(self.device),
                pad_token_id=0,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
            print('output:',output)

            result= self._decode_text(output, tokenizer)
            print('result:',result)

            '''
            output = self.llm.generate(
                inputs_embeds=inputs_embeds,
                pad_token_id=0,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs
            )
            result= self._decode_text(output, tokenizer)
            '''
        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result
       
    # inherit other generation related methods from self.llm[MiniCPMForCausalLM]
    def prepare_inputs_for_generation(**kwargs):
        return self.llm.prepare_inputs_for_generation(**kwargs)
