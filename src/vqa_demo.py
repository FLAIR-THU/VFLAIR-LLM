import argparse
from evaluates.MainTaskVFL_LLM import *
from load.LoadConfigs import *
from load.LoadParty import load_parties_llm
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, AutoProcessor
from utils.qwen_vl_utils import process_vision_info



def get_cls_ancestor(model_type: str = 'qwen2', architecture: str = 'CLM'):
    if architecture == 'MM':
        from models.llm_models.llama import LlamaTailForCausalLM_forMM
        from models.llm_models.minicpmv import MiniCPMVModelTail
        from models.llm_models.minicpm import MiniCPMTailForCausalLM
        from models.llm_models.minigpt4.minigpt4 import MiniGPT4Tail

        MM_MODEL_MAPPING={
            'llama':LlamaTailForCausalLM_forMM, #MiniGPT4Tail, #,
            'minicpm': MiniCPMTailForCausalLM,
            'minicpmv': MiniCPMVModelTail,
            'minigpt4': MiniGPT4Tail
        }
        target_cls = MM_MODEL_MAPPING[model_type] 
    
    else:
        if model_type == 'chatglm':
            from models.llm_models import chatglm
            target_cls = getattr(chatglm, "ChatGLMForConditionalGeneration")
        elif model_type == 'baichuan':
            from models.llm_models import baichuan
            target_cls = getattr(baichuan, "BaiChuanForCausalLM")
        elif model_type == 'deepseek_v3':
            from models.llm_models import deepseek
            target_cls = getattr(deepseek, "DeepseekV3TailForCausalLM")
        elif model_type == 'llama':
            from models.llm_models import llama
            target_cls = getattr(llama, "LlamaTailForCausalLM")
        elif model_type == 't5':
            from models.llm_models import t5
            target_cls = getattr(t5, "T5ForConditionalGenerationTail_3slice")
        elif model_type == 'qwen2_vl':
            from models.llm_models import qwen2_vl
            target_cls = getattr(qwen2_vl, "Qwen2VLTailForConditionalGeneration")
        else:
            
            from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, \
                MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
            target_module = __import__('transformers')
            aa = {"CLM": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
                "MM": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
                "TQA": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
                "CLS": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES}[architecture][model_type]
            target_cls = getattr(target_module, aa)
        
    return target_cls

def create_sl_llm(sl_config_path):
    parser = argparse.ArgumentParser("sl-llm")
    args = parser.parse_args()
    args = load_basic_configs_llm(sl_config_path, args)
    
    args.dataset = args.dataset_split['dataset_name']
    args.device = 'cuda'
    
    args.exp_res_dir = "./"
    args.exp_res_path = "./"
    
    args = load_attack_configs(sl_config_path, args, -1)

    args = load_parties_llm(args, need_data=False)

    # Build Main Task: inherit generation functions from global model
    ancestor_cls = get_cls_ancestor(args.config.model_type, args.model_architect)
    print('ancestor_cls:',ancestor_cls)
    MainTaskVFL_LLM = create_main_task(ancestor_cls)
    
    vfl = MainTaskVFL_LLM(args)
    vfl.init_communication()

    print('Finish Building vfl:',type(vfl))
    return vfl



# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
messages = [
    # Video
    ## Local video path
    {"role": "user", "content": [{"type": "video", "video": "/shared/project/guzx/dark/0.mp4"}, 
                                  {"type": "text", "text": "Describe this video."}]}
]

messages = [
    # Video
    ## Local video path
    {"role": "user", "content": [{"type": "video", "video": "/shared/project/guzx/dark/1.mp4"}, 
                                  {"type": "text", "text": "Describe this video."}]}
]
sl_config_path = '/demo/qwen2vl'
model_path = "/shared/model/Qwen2-VL-2B"
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# sl_llm = Qwen2VLForConditionalGeneration.from_pretrained(model_path, use_cache=False,  device_map="auto")
sl_llm = create_sl_llm(sl_config_path)

# [0]['content']
text = processor.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
# print('text:',text)

images, videos = process_vision_info(messages)
inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors="pt")

inputs = {k: v.to(sl_llm.device) for k, v in inputs.items()}
prev_len = inputs['input_ids'].shape[-1] # [1, len]
print('inputs:',inputs.keys())

generated_ids = sl_llm.generate(**inputs, use_cache=False, max_new_tokens = 4)[:,prev_len:]

answer = tokenizer.batch_decode(generated_ids)
print('answer:',answer)
