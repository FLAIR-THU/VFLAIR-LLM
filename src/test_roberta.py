# from transformers import  GPT2PreTrainedModel# ,GPT2Model
import torch.nn as nn
import torch
import copy
import numpy as np
import random
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel,AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

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

}

current_model_type = "roberta"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[current_model_type])#,trust_remote_code=True)
full_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH[current_model_type])#,trust_remote_code=True) # AutoModelForCausalLM



model_type = 'Roberta'
# for name, param in full_gpt.named_parameters():  GPT2LMHeadModel
#     print("-----full_gpt--{}:{}".format(name, ""))

pad_token = '[PAD]'
tokenizer.pad_token = pad_token # ({'pad_token': '[PAD]'}) # args.tokenizer.eos_token #
pad_id = tokenizer.convert_tokens_to_ids(pad_token)  #
full_model.config.pad_token_id = pad_id
config = full_model.config 

inputs = tokenizer( ["Hello, how are you ?","Are you OK?"], \
                                padding='max_length',  # Pad to max_length
                                # truncation='longest_first',  # Truncate to max_length
                                max_length=12,  
                                return_tensors='pt')
print('inputs:',inputs['input_ids']) #[1,8]

print('-'*25)

print('Full Type')
full_model.eval()
logits = full_model(**inputs)
print('logits:',logits.logits.shape,logits.logits)  # 1,3,vocab_size


