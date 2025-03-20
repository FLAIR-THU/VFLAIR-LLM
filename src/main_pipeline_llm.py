import os
import sys
import numpy as np
import time

import random
import logging
import argparse
import torch
torch.autograd.set_detect_anomaly(True)
from peft.peft_model import PeftModel

from load.LoadConfigs import *  # load_configs load_basic_configs_llm
from load.LoadParty import load_parties, load_parties_llm

from evaluates.MainTaskVFL_LLM import *
from utils.basic_functions import append_exp_res
from utils import recorder

import warnings
warnings.filterwarnings("ignore")
import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2000"

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def evaluate_no_attack_pretrained(args):
    # No Attack
    set_seed(args.current_seed)

    vfl = MainTaskVFL_LLM(args)
    vfl.init_communication()

    exp_result, metric_val = vfl.inference(need_save_state = args.need_final_epoch_state)

    # # Save record 
    exp_result = f"NoAttack|{args.pad_info}|seed={args.current_seed}|K={args.k}" + exp_result
    print(exp_result)
    append_exp_res(args.exp_res_path, exp_result)

    return vfl, metric_val

def evaluate_no_attack_finetune(args):
    # No Attack
    set_seed(args.current_seed)

    vfl = MainTaskVFL_LLM(args)
    vfl.init_communication()

    exp_result, metric_val, training_time = vfl.train_vfl()

    # attack_metric = main_acc_noattack - main_acc
    # attack_metric_name = 'acc_loss'

    # # Save record 
    exp_result = f"NoAttack|{args.pad_info}|finetune={args.finetune_name}|seed={args.current_seed}|K={args.k}|bs={args.batch_size}|LR={args.main_lr}|num_class={args.num_classes}|Q={args.Q}|epoch={args.main_epochs}|early_stop_threshold={args.early_stop_threshold}|headlayer={args.head_layer_trainable}|model_slice_trainable={args.model_slice_trainable}|local_encoders_num={args.local_encoders_num}|local_tail_encoders_num={args.local_tail_encoders_num}|vfl_model_slice_num={args.vfl_model_slice_num}|" \
                 + exp_result
    print(exp_result)

    append_exp_res(args.exp_res_path, exp_result)

    return vfl, metric_val


def evaluate_inversion_attack(args):
    for index in args.inversion_index:
        torch.cuda.empty_cache()
        set_seed(args.current_seed)

        args = load_attack_configs(args.configs, args, index)
        print('======= Test Attack', index, ': ', args.attack_name, ' =======')
        print('attack configs:', args.attack_configs)

        if args.basic_vfl != None:
            vfl = args.basic_vfl
            main_tack_acc = args.main_acc_noattack
        else:
            # args.need_auxiliary = 1
            args = load_parties_llm(args)
            vfl = MainTaskVFL_LLM(args)
            vfl.init_communication()

            if args.pipeline == 'finetune':
                _exp_result, metric_val, training_time = vfl.train_vfl()
            elif args.pipeline == 'pretrained':
                _exp_result, metric_val = vfl.inference(need_save_state = args.need_final_epoch_state)
            main_tack_acc = metric_val
            print(_exp_result)

        print('=== Begin Attack ===')
        training_time = vfl.training_time
        precision, recall , attack_total_time= vfl.evaluate_attack()

        target_data = args.attack_configs['target_data']
        exp_result = f"{args.attack_name}|{args.pad_info}|target_data={target_data}|finetune={args.finetune_name}|seed={args.current_seed}|K={args.k}|bs={args.batch_size}|LR={args.main_lr}|num_class={args.num_classes}|Q={args.Q}|epoch={args.main_epochs}|early_stop_threshold={args.early_stop_threshold}|final_epoch={vfl.final_epoch}|headlayer={args.head_layer_trainable}|model_slice_trainable={args.model_slice_trainable}|local_encoders_num={args.local_encoders_num}|local_tail_encoders_num={args.local_tail_encoders_num}|vfl_model_slice_num={args.vfl_model_slice_num}|main_task_acc={main_tack_acc}|precision={precision}|recall={recall}|training_time={training_time}|attack_time={attack_total_time}|"
        print(exp_result)
        append_exp_res(args.exp_res_path, exp_result)
    return precision, recall

def evaluate_label_inference_attack(args):
    for index in args.label_inference_index:
        torch.cuda.empty_cache()
        set_seed(args.current_seed)

        args = load_attack_configs(args.configs, args, index)
        print('======= Test Attack', index, ': ', args.attack_name, ' =======')
        print('attack configs:', args.attack_configs)

        if args.basic_vfl != None:
            vfl = args.basic_vfl
            main_tack_acc = args.main_acc_noattack
        else:
            # args.need_auxiliary = 1
            args = load_parties_llm(args)
            vfl = MainTaskVFL_LLM(args)
            vfl.init_communication()

            if args.pipeline == 'finetune':
                _exp_result, metric_val, training_time = vfl.train_vfl()
            elif args.pipeline == 'pretrained':
                _exp_result, metric_val = vfl.inference(need_save_state = args.need_final_epoch_state)
            main_tack_acc = metric_val
            print(_exp_result)

        print('=== Begin Attack ===')
        training_time = vfl.training_time

        rec_rate , attack_total_time= vfl.evaluate_attack()
        if isinstance(rec_rate, dict):
            gen_score = rec_rate['gen_score']
            label_score = rec_rate['label_score']
            exp_result = f"{args.attack_name}|{args.pad_info}|finetune={args.finetune_name}|"+\
            f"seed={args.current_seed}|K={args.k}|bs={args.batch_size}|LR={args.main_lr}|"+\
            f"num_class={args.num_classes}|Q={args.Q}|epoch={args.main_epochs}|early_stop_threshold={args.early_stop_threshold}|final_epoch={vfl.final_epoch}|"+\
            f"headlayer={args.head_layer_trainable}|model_slice_trainable={args.model_slice_trainable}|"+\
            f"local_encoders_num={args.local_encoders_num}|local_tail_encoders_num={args.local_tail_encoders_num}|vfl_model_slice_num={args.vfl_model_slice_num}|"+\
            f"main_task_acc={main_tack_acc}|gen_score={gen_score}|label_score={label_score}|"+\
            f"training_time={training_time}|attack_time={attack_total_time}|"
        else:
            exp_result = f"{args.attack_name}|{args.pad_info}|finetune={args.finetune_name}|seed={args.current_seed}|K={args.k}|bs={args.batch_size}|LR={args.main_lr}|num_class={args.num_classes}|Q={args.Q}|epoch={args.main_epochs}|early_stop_threshold={args.early_stop_threshold}|final_epoch={vfl.final_epoch}|headlayer={args.head_layer_trainable}|model_slice_trainable={args.model_slice_trainable}|local_encoders_num={args.local_encoders_num}|local_tail_encoders_num={args.local_tail_encoders_num}|vfl_model_slice_num={args.vfl_model_slice_num}|main_task_acc={main_tack_acc}|rec_rate={rec_rate}|training_time={training_time}|attack_time={attack_total_time}|"
        
        print(exp_result)
        append_exp_res(args.exp_res_path, exp_result)
    # return rec_rate


def get_cls_ancestor(model_type: str = 'qwen2', architecture: str = 'CLM'):
    if architecture == 'MM':
        from src.models.llm_models.llama import LlamaTailForCausalLM_forMM
        from src.models.llm_models.minicpmv import MiniCPMVModelTail
        from src.models.llm_models.minicpm import MiniCPMTailForCausalLM
        from src.models.llm_models.minigpt4.minigpt4 import MiniGPT4Tail

        # from src.load.llm_model_loaders.minigpt4. import MiniGPTBaseTail #
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

def create_exp_dir_and_file(args):
    # dataset, vfl_model_slice_num, split_info, model_name, pipeline, defense_name='', defense_param='',prefix = ''):
    args.model_name = args.model_list["name"]  
    args.split_info = f'{str(args.local_encoders_num)}_{str(args.local_tail_encoders_num)}'
    exp_res_dir = f'exp_result/{args.dataset}/{args.prefix}/{str(args.vfl_model_slice_num)}-slice/{args.split_info}/'
    if not os.path.exists(exp_res_dir):
        os.makedirs(exp_res_dir)
    if args.pipeline == 'pretrained':
        filename = f'{args.defense_name}_{args.defense_param},pretrained_model={args.model_name}.txt'
    else:
        filename = f'{args.defense_name}_{args.defense_param},finetuned_model={args.model_name}.txt'
    exp_res_path = exp_res_dir + str(filename).replace('/', '')
    
    defense_model_folder = get_defense_model_folder(args)
    model_folder, trained_model_folder = get_model_folder(args)
    
    return exp_res_dir, exp_res_path, model_folder, defense_model_folder, trained_model_folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=60, help='random seed')
    parser.add_argument('--prefix', type=str, default="", help='result_file_prefix')
    parser.add_argument('--configs', type=str, default='basic_configs_news', help='configure json file path')
    parser.add_argument('--save_model', type=int, default=0, help='whether to save the trained model')
    parser.add_argument('--save_defense_model', type=bool, default=True, help='whether to save the defense model')
    parser.add_argument('--attack_only', type=int, default=0, help='attack_only')
    
    
    args = parser.parse_args()

    seed_list = [args.seed]
    
    for seed in seed_list:  
        args.current_seed = seed
        set_seed(seed)
        print('================= iter seed ', seed, ' =================')

        

        ####### load configs from *.json files #######
        
        ############ Basic Configs ############
        args = load_basic_configs_llm(args.configs, args)
        args.need_auxiliary = 0  
        if args.device == 'cuda':
            cuda_id = args.gpu
            torch.cuda.set_device(cuda_id)
            print(f'running on cuda{torch.cuda.current_device()}')
        else:
            print('running on cpu')
        assert 'dataset_name' in args.dataset_split, 'dataset not specified, please add the name of the dataset in config json file'
        args.dataset = args.dataset_split['dataset_name']
        print('Dataset:',args.dataset)
        
        print('======= Defense ========')
        print('Defense_Name:', args.defense_name)
        print('Defense_Config:', str(args.defense_configs))
        
        print('===== Total Attack Tested:', args.attack_num, ' ======')
        print('inversion:', args.inversion_list, args.inversion_index)

        # Save record for different defense method
        exp_res_dir, exp_res_path, model_folder, defense_model_folder, trained_model_folder = create_exp_dir_and_file(args)
        args.exp_res_dir = exp_res_dir
        args.exp_res_path = exp_res_path
        args.model_folder = model_folder
        args.defense_model_folder = defense_model_folder
        args.trained_model_folder = trained_model_folder
        print('Experiment Result Path:',args.exp_res_path)
        print('Save Defense Model Path:',args.defense_model_folder)
        print('=================================\n')

        
        args = load_attack_configs(args.configs, args, -1)

        args = load_parties_llm(args)

        # Build Main Task: inherit generation functions from global model
        ancestor_cls = get_cls_ancestor(args.config.model_type, args.model_architect)
        MainTaskVFL_LLM = create_main_task(ancestor_cls)

        
        args.basic_vfl = None
        args.main_acc_noattack = None

        # vanilla
        if args.pipeline == 'pretrained':
            args.basic_vfl, args.main_acc_noattack = evaluate_no_attack_pretrained(args)
        elif args.pipeline == 'finetune':
            args.basic_vfl, args.main_acc_noattack = evaluate_no_attack_finetune(args)

        # with attack
        if args.inversion_list != []:
            evaluate_inversion_attack(args)

        if args.label_inference_list != []:
            evaluate_label_inference_attack(args)

        append_exp_res(args.exp_res_path, f'\n')
        
        logger.info(recorder)
