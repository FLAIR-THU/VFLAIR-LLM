import sys, os

sys.path.append(os.pardir)

import argparse
import numpy as np
import pickle
from transformers import BertTokenizer, GPT2Tokenizer, LlamaTokenizer
from transformers import BertModel, GPT2Model, LlamaModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, \
    AutoModelForCausalLM, AutoModelForQuestionAnswering
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
import torch.nn as nn
import torch
import copy
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftModelForCausalLM


from .llm_model_loaders.BertModelLoader import *
from .llm_model_loaders.RobertaModelLoader import *
from .llm_model_loaders.GPT2ModelLoader import *
from .llm_model_loaders.LlamaModelLoader import *
from .llm_model_loaders.GemmaModelLoader import *
from .llm_model_loaders.FalconModelLoader import *
from .llm_model_loaders.MistralModelLoader import *
from .llm_model_loaders.BaichuanModelLoader import *
from .llm_model_loaders.ChatGLMModelLoader import *
from .llm_model_loaders.MambaModelLoader import *
from .llm_model_loaders.XLNetModelLoader import *
from .llm_model_loaders.Qwen2ModelLoader import *
from .llm_model_loaders.T5ModelLoader import *

from .llm_model_loaders.MiniCPMModelLoader import *
from .llm_model_loaders.MiniCPMVModelLoader import *
from .llm_model_loaders.MiniGPT4ModelLoader import *


from models.bottom_models import *
from models.global_models import *
from models.autoencoder import *
from utils.optimizers import MaliciousSGD, MaliciousAdam
from config import vfl_basic_config

Loader_Map = {
    'Roberta': RobertaModelLoader,
    'Bert': BertModelLoader,
    'GPT2': GPT2ModelLoader,
    'Llama': LlamaModelLoader,
    'llama': LlamaModelLoader,
    'Gemma': GemmaModelLoader,
    'Falcon': FalconModelLoader,
    'Mistral': MistralModelLoader,
    'Baichuan': BaichuanModelLoader,
    'ChatGLM': ChatGLMModelLoader,
    'Mamba': MambaModelLoader,
    'XLNet': XLNetModelLoader,
    'Qwen2': Qwen2ModelLoader,
    'T5':T5ModelLoader,

    'MiniGPT4': MiniGPT4ModelLoader,
    'MiniCPM': MiniCPMModelLoader,
    'MiniCPMV': MiniCPMVModelLoader,

}


YOUR_MODEL_PATH = "/home/DAIR/guzx/Git_FedProject/Models/"
MODEL_PATH = {
    'bert-base-uncased': YOUR_MODEL_PATH + "bert-base-uncased",
    "textattackbert-base-uncased-CoLA": YOUR_MODEL_PATH + "textattackbert-base-uncased-CoLA",
    "textattackbert-base-uncased-yelp-polarity": YOUR_MODEL_PATH + "textattackbert-base-uncased-yelp-polarity",
    "textattackbert-base-uncased-SST-2": YOUR_MODEL_PATH + "textattackbert-base-uncased-SST-2",
    "textattackbert-base-cased-STS-B": YOUR_MODEL_PATH + "textattackbert-base-cased-STS-B",
    "textattackbert-base-uncased-MRPC": YOUR_MODEL_PATH + "textattackbert-base-uncased-MRPC",
    "textattackbert-base-uncased-MNLI": YOUR_MODEL_PATH + "textattackbert-base-uncased-MNLI",
    "textattackbert-base-uncased-QNLI": YOUR_MODEL_PATH + "textattackbert-base-uncased-QNLI",
    "textattackbert-base-uncased-QQP": YOUR_MODEL_PATH + "textattackbert-base-uncased-QQP",
    "textattackbert-base-uncased-WNLI": YOUR_MODEL_PATH + "textattackbert-base-uncased-WNLI",
    "textattackbert-base-uncased-RTE": YOUR_MODEL_PATH + "textattackbert-base-uncased-RTE",
    "textattackroberta-base-CoLA": YOUR_MODEL_PATH + "textattackbert-base-uncased-CoLA",
    "textattackroberta-base-SST-2": YOUR_MODEL_PATH + "textattackroberta-base-SST-2",
    "textattackalbert-base-v2-CoLA": YOUR_MODEL_PATH + "textattackalbert-base-v2-CoLA",
    "textattackroberta-base-MNLI": YOUR_MODEL_PATH + "textattackroberta-base-MNLI",
    "nihaldsouza1yelp-rating-classification": YOUR_MODEL_PATH + "nihaldsouza1yelp-rating-classification",
    "deepsetroberta-base-squad2": YOUR_MODEL_PATH + "deepsetroberta-base-squad2",
    "rsvp-aibertserini-bert-base-squad": YOUR_MODEL_PATH + "rsvp-aibertserini-bert-base-squad",
    "FabianWillnerdistilbert-base-uncased-finetuned-squad": YOUR_MODEL_PATH + "FabianWillnerdistilbert-base-uncased-finetuned-squad",
    "google-bertbert-large-uncased-whole-word-masking-finetuned-squad": YOUR_MODEL_PATH + "google-bertbert-large-uncased-whole-word-masking-finetuned-squad",
    "Shunianyelp_review_classification": YOUR_MODEL_PATH + "Shunianyelp_review_classification",

    "gpt2": YOUR_MODEL_PATH + "gpt2",
    "gpt2-medium": YOUR_MODEL_PATH + "gpt2-medium",
    "gpt2-large": YOUR_MODEL_PATH + "gpt2-large",
    "George-Ogdengpt2-medium-finetuned-mnli": YOUR_MODEL_PATH + "George-Ogdengpt2-medium-finetuned-mnli",
    "michelecafagna26gpt2-medium-finetuned-sst2-sentiment": YOUR_MODEL_PATH + "michelecafagna26gpt2-medium-finetuned-sst2-sentiment",
    "tanzeelabbasGPT-2_fine-tuned_squad_2.0_QA": YOUR_MODEL_PATH + "tanzeelabbasGPT-2_fine-tuned_squad_2.0_QA",

    'llama-2-7b': YOUR_MODEL_PATH + "llama-2-7b",
    'chavinloalpaca-native': YOUR_MODEL_PATH + "chavinloalpaca-native",
    "lqtrung1998Codellama-7b-hf-ReFT-GSM8k": YOUR_MODEL_PATH+"lqtrung1998Codellama-7b-hf-ReFT-GSM8k",
    "meta-mathMetaMath-Mistral-7B" : YOUR_MODEL_PATH + "meta-mathMetaMath-Mistral-7B",

    "HuggingFaceM4tiny-random-LlamaForCausalLM": YOUR_MODEL_PATH + "HuggingFaceM4tiny-random-LlamaForCausalLM",
    "HuggingFaceH4tiny-random-LlamaForSequenceClassification": YOUR_MODEL_PATH + "HuggingFaceH4tiny-random-LlamaForSequenceClassification",

    "AudreyTrungNguyenllama-qnli-p-tuning": YOUR_MODEL_PATH + "AudreyTrungNguyenllama-qnli-p-tuning",

    "googleflan-t5-base": YOUR_MODEL_PATH + "googleflan-t5-base",
    "echarlaixtiny-random-mistral": YOUR_MODEL_PATH + "echarlaixtiny-random-mistral",
    "baichuan-incBaichuan-7B": YOUR_MODEL_PATH + "baichuan-incBaichuan-7B",
    "fxmartytiny-random-GemmaForCausalLM": YOUR_MODEL_PATH + 'fxmartytiny-random-GemmaForCausalLM',
    "state-spacesmamba-130m-hf": YOUR_MODEL_PATH + "state-spacesmamba-130m-hf",
    "THUDMchatglm3-6b": YOUR_MODEL_PATH + "THUDMchatglm3-6b",
    "katuni4katiny-random-chatglm2": YOUR_MODEL_PATH + "katuni4katiny-random-chatglm2",
    "xlnetxlnet-base-cased": YOUR_MODEL_PATH + "xlnetxlnet-base-cased",
    "tiiuaefalcon-rw-1b": YOUR_MODEL_PATH + "tiiuaefalcon-rw-1b",
    "fxmartytiny-random-GemmaForCausalLM": YOUR_MODEL_PATH + "fxmartytiny-random-GemmaForCausalLM",
    "mistralaiMistral-7B-Instruct-v0.2": YOUR_MODEL_PATH + "mistralaiMistral-7B-Instruct-v0.2",

    "googlegemma-2b": YOUR_MODEL_PATH + "googlegemma-2b",
}
MODEL_PATH.update({'Qwen/Qwen1.5-0.5B-Chat': None})

LLM_supported = MODEL_PATH.keys()


def create_model(bottom_model, ema=False, size_bottom_out=10, num_classes=10):
    model = BottomModelPlus(bottom_model, size_bottom_out, num_classes,
                            num_layer=2,
                            activation_func_type='ReLU',
                            use_bn=0)
    model = model

    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def load_models(args):
    args.net_list = [None] * args.k
    for ik in range(args.k):
        current_model_type = args.model_list[str(ik)]['type']
        current_model_path = args.model_list[str(ik)]['path']
        args.net_list[ik] = pickle.load(
            open('.././src/models/model_parameters/' + current_model_type + '/' + current_model_path + '.pkl', "rb"))
        args.net_list[ik] = args.net_list[ik].to(args.device)
    # important
    return args

def load_basic_models(args, index):
    current_model_type = args.model_list[str(index)]['type']
    print(f"current_model_type={current_model_type}")
    current_input_dim = args.model_list[str(index)]['input_dim'] if 'input_dim' in args.model_list[str(index)] else -1
    current_hidden_dim = args.model_list[str(index)]['hidden_dim'] if 'hidden_dim' in args.model_list[
        str(index)] else -1
    current_output_dim = args.model_list[str(index)]['output_dim'] if 'output_dim' in args.model_list[
        str(index)] else -1
    current_vocab_size = args.model_list[str(index)]['vocab_size'] if 'vocab_size' in args.model_list[
        str(index)] else -1
    # print(f"index={index}, current_input_dim={current_input_dim}, current_output_dim={current_output_dim}")
    # current_model_path = args.model_list[str(index)]['path']
    # local_model = pickle.load(open('.././model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
    if 'resnet' in current_model_type.lower() or 'lenet' in current_model_type.lower() or 'cnn' in current_model_type.lower() or 'alexnet' in current_model_type.lower():
        local_model = globals()[current_model_type](current_output_dim)
    elif 'gcn' in current_model_type.lower():
        local_model = globals()[current_model_type](nfeat=current_input_dim, nhid=current_hidden_dim,
                                                    nclass=current_output_dim, device=args.device,
                                                    dropout=0.0, lr=args.main_lr)
    elif 'lstm' in current_model_type.lower():
        local_model = globals()[current_model_type](current_vocab_size, current_output_dim)
    else:
        local_model = globals()[current_model_type](current_input_dim, current_output_dim)
    local_model = local_model.to(args.device)
    print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")
    local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)
    # print(f"use SGD for local optimizer for PMC checking")
    # local_model_optimizer = torch.optim.SGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=5e-4)

    # update optimizer
    if 'activemodelcompletion' in args.attack_name.lower() and index in args.attack_configs['party']:
        print('AMC: use Malicious optimizer for party', index)
        # local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)     
        # local_model_optimizer = MaliciousSGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=5e-4)
        local_model_optimizer = MaliciousAdam(list(local_model.parameters()), lr=args.main_lr)

    global_model = None
    global_model_optimizer = None
    if index == args.k - 1:
        if args.apply_trainable_layer == 0:
            global_model = globals()[args.global_model]()
            global_model = global_model.to(args.device)
            global_model_optimizer = None
        else:
            print("global_model", args.global_model)
            global_input_dim = 0
            for ik in range(args.k):
                global_input_dim += args.model_list[str(ik)]['output_dim']
            global_model = globals()[args.global_model](global_input_dim, args.num_classes)
            global_model = global_model.to(args.device)
            global_model_optimizer = torch.optim.Adam(list(global_model.parameters()), lr=args.main_lr)
            # print(f"use SGD for global optimizer for PMC checking")
            # global_model_optimizer = torch.optim.SGD(list(global_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=5e-4)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer

def load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer):
    print('Load Defense models')
    # no defense at all, set some variables as None
    args.encoder = None
    # some defense need model, add here
    if args.apply_defense == True:
        current_bottleneck_scale = int(args.defense_configs['bottleneck_scale']) if 'bottleneck_scale' in args.defense_configs else 1
        
        if 'std_shift_hyperparameter' in args.defense_configs:
            std_shift_hyperparameter = int(args.defense_configs['std_shift_hyperparameter'])
        else:
            std_shift_hyperparameter = 5 if ('mnist' in args.dataset.lower() or 'nuswide' == args.dataset.lower() or 'cifar' in args.dataset.lower()) else 0.5 

        if 'MID' in args.defense_name.upper():
            if not 'party' in args.defense_configs:
                args.defense_configs['party'] = [args.k-1]
                print('[warning] default active party selected for applying MID')
            if not 'lambda' in args.defense_configs:
                args.defense_configs['lambda'] = 0.001
                print('[warning] default hyper-parameter lambda selected for applying MID')
            if not ('lr' in args.defense_configs):
                mid_lr = args.main_lr  
                print('[warning] default hyper-parameter mid_lr selected for applying MID')
            else :
                mid_lr = args.defense_configs['lr'] 
            
            print(f"mid defense parties: {args.defense_configs['party']}")
            if index in args.defense_configs['party']:
                print(f"begin to load mid model for party {index}")
                if index == args.k-1:
                    print(f"load global mid model for party {index},std_shift_hyperparameter={std_shift_hyperparameter}")
                    # add args.k-1 MID model at active party with global_model
                    if 'nuswide' in args.dataset.lower() or 'nus-wide' in args.dataset.lower():
                        print(f"small MID model for nuswide")
                        mid_model_list = [MID_model_small(args.model_list[str(_ik)]['output_dim'],args.model_list[str(_ik)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter) for _ik in range(args.k-1)]
                    else:
                        mid_model_list = [MID_model(args.model_list[str(_ik)]['output_dim'],args.model_list[str(_ik)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter) for _ik in range(args.k-1)]
                    mid_model_list = [model.to(args.device) for model in mid_model_list]
                    global_model = Active_global_MID_model(global_model,mid_model_list)
                    global_model = global_model.to(args.device)
                    # update optimizer
                    if args.apply_trainable_layer == 0:
                        parameters = []
                        for mid_model in global_model.mid_model_list:
                            parameters += list(mid_model.parameters())
                        global_model_optimizer = torch.optim.Adam(parameters, lr=mid_lr)
                        print(f"mid_lr = {mid_lr}")
                    else:
                        parameters = []
                        for mid_model in global_model.mid_model_list:
                            parameters += list(mid_model.parameters())
                        global_model_optimizer = torch.optim.Adam(
                            [{'params': global_model.global_model.parameters(), 'lr': args.main_lr},              
                            {'params': parameters, 'lr': mid_lr}])
                        print(f"mid_lr = {mid_lr}")
                        
                else:
                    print(f"load local mid model for party {index}")
                    # add MID model at passive party with local_model
                    print('lambda for passive party local mid model:',args.defense_configs['lambda'])
                    if 'nuswide' in args.dataset.lower() or 'nus-wide' in args.dataset.lower():
                        print(f"small MID model for nuswide")
                        mid_model = MID_model_small(args.model_list[str(index)]['output_dim'],args.model_list[str(index)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter)
                    else:
                        mid_model = MID_model(args.model_list[str(index)]['output_dim'],args.model_list[str(index)]['output_dim'],args.defense_configs['lambda'],bottleneck_scale=current_bottleneck_scale, std_shift=std_shift_hyperparameter)
                    mid_model = mid_model.to(args.device)
                    local_model = Passive_local_MID_model(local_model,mid_model)
                    local_model = local_model.to(args.device)

                    # update optimizer
                    if 'activemodelcompletion' in args.attack_name.lower() and index in args.attack_configs['party']:
                        print('AMC: use Malicious optimizer for party', index)
                        # local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.main_lr, weight_decay=0.0)     
                        # local_model_optimizer = MaliciousSGD(list(local_model.parameters()), lr=args.main_lr, momentum=0.0, weight_decay=5e-4)
                        # local_model_optimizer = MaliciousAdam(list(local_model.parameters()),lr=args.main_lr)
                        local_model_optimizer = MaliciousAdam(
                            [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},              
                            {'params': local_model.mid_model.parameters(), 'lr': mid_lr}])
                        # assert 1>2
                    else:
                        local_model_optimizer = torch.optim.Adam(
                            [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},              
                            {'params': local_model.mid_model.parameters(), 'lr': mid_lr}])

        if 'adversarial' in args.defense_name.lower(): # for adversarial training
            # add adversarial model for local model
            if not 'party' in args.defense_configs:
                args.defense_configs['party'] = [0]
                print('[warning] default passive party selected for applying adversarial training')
            if not ('lr' in args.defense_configs):
                adversarial_lr = args.main_lr  
                print('[warning] default hyper-parameter mid_lr selected for applying MID')
            else :
                adversarial_lr = args.defense_configs['lr']
            if not ('model' in args.defense_configs):
                model_name = 'Adversarial_MLP2'
            else:
                model_name = args.defense_configs['model']
            print(model_name)
            if index in args.defense_configs['party']:
                # assert args.parties[index].train_attribute != None, "[Error] no attribute for adversarial"
                # add adversarial model to the the defense party=index
                adversarial_input_dim = args.model_list[str(index)]['output_dim']
                adversarial_output_dim = args.num_attributes
                # print(f"[debug] in load defense model, adversarial_input_dim={adversarial_input_dim}, adversarial_output_dim={adversarial_output_dim}")
                adversarial_model = globals()[model_name](adversarial_input_dim, adversarial_output_dim)
                local_model = Local_Adversarial_combined_model(local_model,adversarial_model)
                local_model = local_model.to(args.device)
                # update optimizer
                local_model_optimizer = torch.optim.Adam(
                            [{'params': local_model.local_model.parameters(), 'lr': args.main_lr},              
                            {'params': local_model.adversarial_model.parameters(), 'lr': adversarial_lr}])
            
        if 'CAE' in args.defense_name.upper(): # for CAE and DCAE
            # print("CAE in defense_name,", args.defense_name)
            if index == args.k-1:
                # only active party can have encoder and decoder for CAE
                assert 'model_path' in args.defense_configs, "[error] no CAE model path given"
                if not 'input_dim' in args.defense_configs:
                    args.defense_configs['input_dim'] = args.num_classes
                    print('[warning] default input_dim selected as num_classes for applying CAE')
                if not 'encode_dim' in args.defense_configs:
                    args.defense_configs['encode_dim'] = 2 + 6 * args.defense_configs['input_dim']
                    print('[warning] default encode_dim selected as 2+6*input_dim for applying CAE')
                if args.num_classes > 20:
                    encoder = AutoEncoder_large(real_dim=args.defense_configs['input_dim'], input_dim=20, encode_dim=args.defense_configs['encode_dim']).to(args.device)
                else:
                    encoder = AutoEncoder(input_dim=args.defense_configs['input_dim'], encode_dim=args.defense_configs['encode_dim']).to(args.device)
                encoder.load_model(args.defense_configs['model_path'], target_device=args.device)
                args.encoder = encoder
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer

def load_models_per_party(args, index):
    current_model_type = args.model_list[str(index)]['type']
    val_model = None
    args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models(args, index)
    args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_defense_models(args,index,local_model,local_model_optimizer,global_model,global_model_optimizer)
    # important
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer


def load_llm_slice(args, slice_index):
    assert args.model_type in Loader_Map.keys(), f'{args.model_type} not supported'
    # is_active_party = (index == args.k - 1)
    loader = Loader_Map[args.model_type]()
    model_path = args.model_path[0]
    result = loader.load_slice(args=args, model_path=model_path, slice_index = slice_index)
    return result

def load_basic_models_llm(args, index):
    assert args.model_type in Loader_Map.keys(), f'{args.model_type} not supported'
    is_active_party = (index == args.k - 1)
    loader = Loader_Map[args.model_type]()
    model_path = args.model_path[index]
    result = loader.load(args=args, model_path=model_path, is_active_party = is_active_party)
    return result
'''
return {
    "models": self._models,
    "config": model_config,
    "generation_config": generation_config,
    "model_architectures": model_architectures,
    "model_embedded_dim": model_embedded_dim,
    "all_encoders_num": all_encoders_num,
    "model_dtype": model_dtype
}
'''

def load_models_per_party_llm(args, index):
    result = load_basic_models_llm(args,index)
    return result


if __name__ == '__main__':
    pass
