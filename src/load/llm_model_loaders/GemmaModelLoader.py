from .LLMModelLoader import LLMModelLoader
from transformers import PreTrainedModel, AutoTokenizer, AutoConfig
from models.llm_models.gemma import ModelPartitionPipelineGemma
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftModelForCausalLM


class GemmaModelLoader(LLMModelLoader):
    _models = {}  # type:dict[int,PreTrainedModel]

    def load(self, args, model_path, is_active_party):
        if args.vfl_model_slice_num == 2:
            split_index = (args.local_encoders_num, )
        elif args.vfl_model_slice_num == 3:
            split_index = (args.local_encoders_num, args.local_tail_encoders_num)
        else:
            raise ValueError(f"Not supported vfl_model_slice_num:{args.vfl_model_slice_num}") 
        
        model_config = AutoConfig.from_pretrained(model_path) # full model config
        if hasattr(model_config, 'generation_config'):
            generation_config = model_config.generation_config
        else:
            generation_config = None
        model_architectures = model_config.architectures
        model_embedded_dim = model_config.hidden_size # change with model type
        all_encoders_num = model_config.num_hidden_layers # change with model type


        p = ModelPartitionPipelineGemma(args=args, all_layer_num = all_encoders_num, 
                            split_index=split_index, is_server=is_active_party)
        self._models=p.from_pretrained(model_path)# **vfl_basic_config.kwargs_model_loading))
        print(f'===== is_active_party={is_active_party}---{self._models.keys()} ======')


        if args.finetune_name == "LoRA":
            print('==== LoRA ====')
            for i, m in self._models.items():
                peft_model = self._set_peft(m, args.finetune_detail_configs)
                self._models.update({i: peft_model})
            for _key in self._models.keys():
                print(_key)
                self._models[_key].print_trainable_parameters()

        if not is_active_party:
            model_head_embedding_trainable = args.embedding_trainable
            if not model_head_embedding_trainable: # freeze embeddings that's not needed
                for param in self._models[0].wte.parameters():
                    param.requires_grad = False
                for param in self._models[0].wpe.parameters():
                    param.requires_grad = False
            model_head_encoder_trainable_ids = args.encoder_trainable_ids['head']
            for encoder_id in range(len(self._models[0].layers)):
                if encoder_id not in model_head_encoder_trainable_ids: # freeze encoders that's not needed
                    for param in self._models[0].layers.parameters():
                        param.requires_grad = False
            print(f'passive_model_head: encoder_trainable_ids={model_head_encoder_trainable_ids}; embedding_trainable={model_head_embedding_trainable}')

            if args.vfl_model_slice_num == 3:
                model_tail_encoder_trainable_ids = args.encoder_trainable_ids['tail']
                print('1 ',type(self._models[2]))
                print('2 ',type(self._models[2].model))
                for encoder_id in range(len(self._models[2].model.layers)):
                    if encoder_id not in model_tail_encoder_trainable_ids: # freeze encoders that's not needed
                        for param in self._models[2].model.layers.parameters():
                            param.requires_grad = False
                model_tail_head_layer_trainable = args.head_layer_trainable
                if not model_tail_head_layer_trainable: # freeze embeddings that's not needed
                    for param in self._models[2].head_layer.parameters():
                        param.requires_grad = False
                print(f'passive_model_tail: encoder_trainable_ids={model_tail_encoder_trainable_ids}; head_layer_trainable={model_tail_head_layer_trainable}', )
        else:
            if args.vfl_model_slice_num == 3:
                model_body_encoder_trainable_ids = args.encoder_trainable_ids['body']
                for encoder_id in range(len(self._models[1].layers)):
                    if encoder_id not in model_body_encoder_trainable_ids: # freeze encoders that's not needed
                        for param in self._models[1].layers.parameters():
                            param.requires_grad = False
                print(f'active_model_body: encoder_trainable_ids={model_body_encoder_trainable_ids}')
                
            else:
                model_tail_encoder_trainable_ids = args.encoder_trainable_ids['tail']
                for encoder_id in range(len(self._models[1].model.layers)):
                    if encoder_id not in model_tail_encoder_trainable_ids: # freeze encoders that's not needed
                        for param in self._models[1].model.layers.parameters():
                            param.requires_grad = False
                model_tail_head_layer_trainable = args.head_layer_trainable
                if not model_tail_head_layer_trainable: # freeze embeddings that's not needed
                    for param in self._models[1].head_layer.parameters():
                        param.requires_grad = False
                print(f'active_model_tail: encoder_trainable_ids={model_tail_encoder_trainable_ids}; head_layer_trainable={model_tail_head_layer_trainable}')


        print('final trainable param:')
        for _key in self._models.keys():
            print(_key)
            self._models[_key].print_trainable_parameters()

        model_dtype = self._get_model_dtype(model_config)


        return {
            "models": self._models,
            "config": model_config,
            "generation_config": generation_config,
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

