from .LLMModelLoader import LLMModelLoader
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
from models.llm_models.minicpmv import ModelPartitionPipelineMiniCPMV
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftModelForCausalLM
from models.llm_models.minicpm import *
from models.llm_models.minicpmv import *

class MiniCPMVModelLoader(LLMModelLoader):
    _models = {}  # type:dict[int,PreTrainedModel]

    def load(self, args, model_path, is_active_party, party_idx):
        if is_active_party:
            print(f'== Load Active Party Model From:{model_path}')
        else:
            print(f'== Load Passive Party Model From:{model_path}')

        if args.vfl_model_slice_num == 2:
            split_index = (args.local_encoders_num, )
        elif args.vfl_model_slice_num == 3:
            split_index = (args.local_encoders_num, args.local_tail_encoders_num)
        else:
            raise ValueError(f"Not supported vfl_model_slice_num:{args.vfl_model_slice_num}") 
        
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True) # full model config

        model_architectures = model_config.architectures
        model_embedded_dim = model_config.hidden_size # change with model type
        all_encoders_num = model_config.num_hidden_layers # change with model type

        p = ModelPartitionPipelineMiniCPMV(args=args, all_layer_num = all_encoders_num, 
                            split_index=split_index, is_server=is_active_party)
        self._models = p.from_pretrained(model_path, **args.kwargs_model_loading)
        generation_config = None

        # print('all_encoders_num:',all_encoders_num)
        # for _key in self._models.keys():
        #     print(f'{_key} num_hidden_layers:{self._models[_key].config.num_hidden_layers}')
        #     print(f'{_key} num_all_hidden_layers:{self._models[_key].config.num_all_hidden_layers}')

        if args.finetune_name == "LoRA":
            print(f'LoRA Configs:{args.finetune_detail_configs}')
            for i, m in self._models.items():
                if not (i == 2 and args.local_tail_encoders_num == 0):
                    peft_model = self._set_peft(m, args.finetune_detail_configs)
                    self._models.update({i: peft_model})
            print('after lora trainable param:')
            for _key in self._models.keys():
                print(_key)
                self._models[_key].print_trainable_parameters()


        model_trainable_info = args.model_trainable_info[party_idx]

        if not is_active_party:
            if not model_trainable_info.model_slice_trainable[0]:
                model_head_vision_processor_trainable = model_trainable_info.vision_processor_trainable
                if not model_head_vision_processor_trainable: # freeze vpm 
                    for param in self._models[0].vpm.parameters():
                        param.requires_grad = False
                    # for param in self._models[0].transform.parameters():
                    #     param.requires_grad = False
                    for param in self._models[0].resampler.parameters():
                        param.requires_grad = False
                    
                model_head_embedding_trainable = model_trainable_info.embedding_trainable
                if not model_head_embedding_trainable: # freeze embeddings 
                    for param in self._models[0].llm.model.embed_tokens.parameters():
                        param.requires_grad = False
                    
                model_head_encoder_trainable_ids = model_trainable_info.encoder_trainable_ids['head']
                for encoder_id in range(len(self._models[0].llm.model.layers)):
                    if encoder_id not in model_head_encoder_trainable_ids: # freeze encoders that's not needed
                        for param in self._models[0].llm.model.layers[encoder_id].parameters():
                            param.requires_grad = False
                print(f'passive_model_head: encoder_trainable_ids={model_head_encoder_trainable_ids}; embedding_trainable={model_head_embedding_trainable}')
            else:
                print(f'passive_model_head: all trainable')

            if args.vfl_model_slice_num == 3:
                if not model_trainable_info.model_slice_trainable[2]:
                    model_tail_encoder_trainable_ids = model_trainable_info.encoder_trainable_ids['tail']
                    for encoder_id in range(len(self._models[2].llm.model.layers)):
                        if encoder_id not in model_tail_encoder_trainable_ids: # freeze encoders that's not needed
                            for param in self._models[2].llm.model.layers[encoder_id].parameters():
                                param.requires_grad = False
                    model_tail_head_layer_trainable = model_trainable_info.head_layer_trainable
                    if not model_tail_head_layer_trainable: # freeze embeddings that's not needed
                        for param in self._models[2].llm.head_layer.parameters():
                            param.requires_grad = False
                        for param in self._models[2].llm.model.norm.parameters():
                            param.requires_grad = False
                    print(f'passive_model_tail: encoder_trainable_ids={model_tail_encoder_trainable_ids}; head_layer_trainable={model_tail_head_layer_trainable}', )
                else:
                    print(f'passive_model_tail: all trainable')
        else:
            if args.vfl_model_slice_num == 3:
                if not model_trainable_info.model_slice_trainable[1]:
                    model_body_encoder_trainable_ids = model_trainable_info.encoder_trainable_ids['body']
                    for encoder_id in range(len(self._models[1].llm.model.layers)):
                        if encoder_id not in model_body_encoder_trainable_ids: # freeze encoders that's not needed
                            for param in self._models[1].llm.model.layers[encoder_id].parameters():
                                param.requires_grad = False
                    print(f'active_model_body: encoder_trainable_ids={model_body_encoder_trainable_ids}')
                else:
                    print(f'active_model_body: all trainable')
            else:
                if not model_trainable_info.model_slice_trainable[1]:
                    model_tail_encoder_trainable_ids = model_trainable_info.encoder_trainable_ids['tail']
                    for encoder_id in range(len(self._models[1].llm.model.layers)):
                        if encoder_id not in model_tail_encoder_trainable_ids: # freeze encoders that's not needed
                            for param in self._models[1].llm.model.layers[encoder_id].parameters():
                                param.requires_grad = False
                    model_tail_head_layer_trainable = model_trainable_info.head_layer_trainable
                    if not model_tail_head_layer_trainable: # freeze embeddings that's not needed
                        for param in self._models[1].llm.head_layer.parameters():
                            param.requires_grad = False
                        for param in self._models[1].llm.model.norm.parameters():
                            param.requires_grad = False
                    print(f'active_model_tail: encoder_trainable_ids={model_tail_encoder_trainable_ids}; head_layer_trainable={model_tail_head_layer_trainable}')
                else:
                    print(f'active_model_tail: all trainable')


        for _key in self._models.keys():
            self._models[_key].print_trainable_parameters()
            for name, param in self._models[_key].named_parameters():
                if param.requires_grad:
                    print(name)

        model_dtype = self._get_model_dtype(model_config)
        # print('_get_model_dtype:',model_dtype)

        for _key in self._models.keys():
            self._models[_key].to(args.device)
            print(f'final load -- model {_key}:{type(self._models[_key])}')
            # if int(_key) == 0:
            #     for name, param in self._models[_key].named_parameters():
            #         print(f"Parameter: {name}, Device: {param.device}")

        return {
            "models": self._models,
            "config": model_config,
            # "generation_config": generation_config,
            "model_architectures": model_architectures,
            "model_embedded_dim": model_embedded_dim,
            "all_encoders_num": all_encoders_num,
            "model_dtype": model_dtype
        }

    def _set_peft(self, model, finetune_detail_configs):
        """
        peft training or load trained peft weights
        :return:
        """
        # print('args.finetune_detail_configs:',args.finetune_detail_configs)
        if finetune_detail_configs != None and finetune_detail_configs!={}:
            lora_config = LoraConfig(
                **finetune_detail_configs
            )
            # print('finetune_detail_configs:',finetune_detail_configs)
        else:
            lora_config = LoraConfig(
                inference_mode=False,  
                r=4,  
                lora_alpha=32, 
                lora_dropout=0.1
            )
            # print('default lora configs')

        def get_lora_model(model):
            model.enable_input_require_grads()
            peft_model = get_peft_model(model, lora_config)
            return peft_model

        model = get_lora_model(model)
        return model

    def _prepare_model_update_args(self):
        model = None
        for m in self._models.values():
            if m:
                model = m

        if model is not None:
            return model.config, model.generation_config
        return None, None

