# from transformers import  GPT2PreTrainedModel# ,GPT2Model
import torch.nn as nn
import torch
import copy
import numpy as np
import random
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel,AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, T5ForConditionalGeneration

def set_seed(seed=0):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(0)

YOUR_MODEL_PATH = "/home/DAIR/guzx/Git_FedProject/Models/"
MODEL_PATH = {
    "googleflan-t5-base": YOUR_MODEL_PATH + "googleflan-t5-base",
    "baichuan-incBaichuan-7B": YOUR_MODEL_PATH + "baichuan-incBaichuan-7B",
    "roberta": "/shared/model/deepsetroberta-base-squad2",
    "t5":"/shared/model/googleflan-t5-small"

}

current_model_type = "t5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[current_model_type])#,trust_remote_code=True)
full_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH[current_model_type])#,trust_remote_code=True) # AutoModelForCausalLM
print('Full Model Type:',type(full_model))

pad_token = '[PAD]'
tokenizer.pad_token = pad_token # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
pad_id = tokenizer.convert_tokens_to_ids(pad_token)  #
full_model.config.pad_token_id = pad_id
config = full_model.config 

inputs = tokenizer( "summarize: studies have shown that owning a dog is good for you.", \
                                # padding='max_length',  # Pad to max_length
                                # truncation='longest_first',  # Truncate to max_length
                                # max_length=12,  
                                return_tensors='pt')
print('inputs.input_ids:',inputs['input_ids']) #[1,8]

print('-'*25)

print('Full Type')
full_model.eval()
outputs = full_model.generate(inputs.input_ids, max_new_tokens = 2, use_cache = False)
print('outputs:',outputs.shape,outputs)  # 1,3,vocab_size

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

