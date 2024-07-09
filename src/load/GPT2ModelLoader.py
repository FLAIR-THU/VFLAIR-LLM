from .IModelLoader import IModelLoader
from transformers import PreTrainedModel, AutoTokenizer
from config import vfl_basic_config
from models.llm_models.gpt2_new import VFLPipelineGPT2
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftModelForCausalLM


class GPT2ModelLoader(IModelLoader):
    _models = {}  # type:dict[int,PreTrainedModel]

    def load(self, args, model_path, is_active_party):
        if args.vfl_model_slice_num == 2:
            split_index = (args.local_encoders_num, )
        elif args.vfl_model_slice_num == 3:
            split_index = (args.local_encoders_num, args.local_tail_encoders_num)
        else:
            raise ValueError(f"Not supported vfl_model_slice_num:{args.vfl_model_slice_num}") 
        # print(f'split_index: {split_index}')
        p = VFLPipelineGPT2(split_index=split_index, is_server=is_active_party, device = args.device)
        self._models.update(p.from_pretrained(model_path))
        # **vfl_basic_config.kwargs_model_loading))
        # self._tensor_to_device(self._models, args.device)

        # print(f'self._models {is_active_party}:{self._models.keys()}')
        # for _key in self._models.keys():
        #     print(f'model {_key} device:',self._models[_key].device)
            
        config, generation_config = self._prepare_model_update_args()
        model_architectures = config.architectures
        model_embedded_dim = config.n_embd
        all_encoders_num = config.n_layer


        if args.finetune_name == "LoRA":
            for i, m in self._models.items():
                peft_model = self._set_peft(m)
                self._models.update({i: peft_model})
            # print('after lora trainable param:')
            # self._models[0].print_trainable_parameters()

        if not is_active_party:
            model_head_embedding_trainable = args.embedding_trainable
            print('model_head embedding_trainable = ', model_head_embedding_trainable)
            if not model_head_embedding_trainable: # freeze embeddings that's not needed
                for param in self._models[0].wte.parameters():
                    param.requires_grad = False
                for param in self._models[0].wpe.parameters():
                    param.requires_grad = False
            model_head_encoder_trainable_ids = args.encoder_trainable_ids['head']
            print('model_head encoder_trainable_ids = ', model_head_encoder_trainable_ids)
            for encoder_id in range(len(self._models[0].h)):
                if encoder_id not in model_head_encoder_trainable_ids: # freeze encoders that's not needed
                    for param in self._models[0].h.parameters():
                        param.requires_grad = False
            
            if args.vfl_model_slice_num == 3:
                model_tail_encoder_trainable_ids = args.encoder_trainable_ids['tail']
                print('model_tail encoder_trainable_ids = ', model_tail_encoder_trainable_ids)
                for encoder_id in range(len(self._models[2].transformer.h)):
                    if encoder_id not in model_tail_encoder_trainable_ids: # freeze encoders that's not needed
                        for param in self._models[2].transformer.h.parameters():
                            param.requires_grad = False

        else:
            model_body_encoder_trainable_ids = args.encoder_trainable_ids['body']
            print('model_body encoder_trainable_ids = ', model_body_encoder_trainable_ids)
            if args.vfl_model_slice_num == 3:
                for encoder_id in range(len(self._models[1].h)):
                    if encoder_id not in model_body_encoder_trainable_ids: # freeze encoders that's not needed
                        for param in self._models[1].h.parameters():
                            param.requires_grad = False
            else:
                for encoder_id in range(len(self._models[1].transformer.h)):
                    if encoder_id not in model_body_encoder_trainable_ids: # freeze encoders that's not needed
                        for param in self._models[1].transformer.h.parameters():
                            param.requires_grad = False

        # print('after config trainable param:')
        # self._models[0].print_trainable_parameters()

        return {
            # "tokenizer": tokenizer,
            "models": self._models,
            "config": config,
            "generation_config": generation_config,
            "model_architectures": model_architectures,
            "model_embedded_dim": model_embedded_dim,
            "all_encoders_num": all_encoders_num
        }

    def _set_peft(self, model):
        """
        peft training or load trained peft weights
        :return:
        """
        # print('args.finetune_detail_configs:',args.finetune_detail_configs)
        if args.finetune_detail_configs != None:
            lora_config = LoraConfig(
                **args.finetune_detail_configs
            )
        else:
            lora_config = LoraConfig(
                inference_mode=False,  
                r=4,  
                lora_alpha=32, 
                lora_dropout=0.1
            )

        def get_lora_model(model):
            model.enable_input_require_grads()
            peft_model = get_peft_model(model, lora_config)
            return peft_model

        model = get_lora_model(model)
        print('after lora')
        model.print_trainable_parameters()
        return model

    def _prepare_model_update_args(self):
        model = None
        for m in self._models.values():
            if m:
                model = m

        if model is not None:
            return model.config, model.generation_config
        return None, None

    def _tensor_to_device(self, dict_like:dict, device):
        for k,v in dict_like.items():
            if isinstance(v,torch.Tensor):
                dict_like[k] = v.to(device)
