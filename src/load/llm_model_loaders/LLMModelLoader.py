from abc import ABCMeta, abstractmethod


class LLMModelLoader(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def load(self, path: str, is_active_party: bool):
        pass

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
