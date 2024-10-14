import sys, os
sys.path.append(os.pardir)
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import numpy as np
import copy
import pickle 
import matplotlib.pyplot as plt
import itertools 
from scipy import optimize
import cv2

from evaluates.attacks.attacker import Attacker
from models.global_models import *
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from dataset.party_dataset import *

from evaluates.defenses.defense_functions import LaplaceDP_for_pred,GaussianDP_for_pred
from models.llm_models.processors.blip_processors import *

def label_to_one_hot(target, num_classes=10):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        # print("use target itself", target.size())
        onehot_target = target.type(torch.float32)
    except:
        target = torch.unsqueeze(target, 1)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes)
        onehot_target.scatter_(1, target, 1)
    return onehot_target

class custom_AE(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super(custom_AE,self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim, 600), 
            nn.LayerNorm(600),
            nn.ReLU(),
            
            nn.Linear(600, 200), 
            nn.LayerNorm(200),
            nn.ReLU(),
            
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            
            nn.Linear(100, target_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.tensor(x,dtype=torch.float32)
        return self.net(x)


class DLG_LLM(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        # 
        self.attack_name = "DLG_LLM"
        self.args = args
        self.top_vfl = top_vfl
        self.vfl_info = top_vfl.final_state

        # self.vfl_first_epoch = top_vfl.first_epoch_state
        
        # prepare parameters
        self.task_type = args.task_type

        self.device = args.device
        self.num_classes = args.num_classes
        self.label_size = args.num_classes
        self.k = args.k
        self.batch_size = args.batch_size

        # attack configs
        self.party = args.attack_configs['party'] # parties that launch attacks , default 1(active party attack)
        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.attack_batch_size = args.attack_configs['batch_size']
        self.attack_sample_num = args.attack_configs['attack_sample_num']
        
        if 'loss_type' not in args.attack_configs.keys():
            self.loss_type = 'cross_entropy'
        else:
            self.loss_type = args.attack_configs['loss_type'] # mse/cross_entropy
        
        if self.loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_type == 'mse':
            self.criterion = nn.MSELoss()
   
    
    def set_seed(self,seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def cal_loss(self, pred, label):

        logits = pred.logits  # [bs, seq_len, vocab_size]
        labels = torch.tensor(label).to(logits.device) # [bs, seq_len]
        # print(f'cal loss logits={logits.shape} labels={labels.shape}')
        if len(labels.shape) > 1:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.args.model_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # print('cal loss:',loss)
        else:
            next_token_logits = logits[:, -1, :]
            labels = torch.tensor(labels.long()).to(self.args.device)
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(next_token_logits, labels)
            # print('loss:', loss)

        return loss


    def attack(self):
        self.set_seed(123)
        print_every = 1

        for attacker_ik in self.party: # attacker party #attacker_ik
            assert attacker_ik == (self.k - 1), 'Only Active party launch input inference attack'

            attacked_party_list = [ik for ik in range(self.k)]
            attacked_party_list.remove(attacker_ik)
            index = attacker_ik
            attack_result = pd.DataFrame(columns = ['img_size','Img_Mse','Rand_Img_Mse'])


            # collect necessary information
            active_model_body_params = list(filter(lambda x: x.requires_grad, self.top_vfl.parties[1].global_model.parameters()))
            # print('active_model_body_params:',len(active_model_body_params))

            vis_processor = self.top_vfl.parties[0].vis_processors['eval']
            #self.vfl_info['vis_processors']['eval']
            reverse_vis_processor = MiniCPM_Transform_reverse()
            
            # attack dst
            test_data = self.vfl_info["test_data"][0] 
            test_label = self.vfl_info["test_label"][0] 
            
            if len(test_data) > self.attack_sample_num:
                test_data = test_data[:self.attack_sample_num]
                test_label = test_label[:self.attack_sample_num]

            if self.args.dataset == 'Lambada' or self.args.dataset == 'Lambada_test':
                attack_test_dataset = LambadaDataset_LLM(self.args, test_data, test_label, 'test')
            elif self.args.dataset == 'TextVQA' or self.args.dataset == 'TextVQA-test':
                attack_test_dataset = TextVQADataset_train(self.args, test_data, test_label, vis_processor,'train')
            elif self.args.dataset == 'GMS8K' or self.args.dataset == 'GMS8K-test':
                attack_test_dataset = GSMDataset_LLM(self.args, test_data, test_label, 'test')
            else:
                attack_test_dataset = PassiveDataset_LLM(self.args, test_data, test_label)

            attack_info = f'Attack Sample Num:{len(attack_test_dataset)}'
            print(attack_info)
            append_exp_res(self.args.exp_res_path, attack_info)

            test_data_loader = DataLoader(attack_test_dataset, batch_size=self.attack_batch_size ,collate_fn=lambda x:x ) # ,
            del(self.vfl_info)
            # test_data_loader = self.vfl_info["test_loader"][0] # Only Passive party has origin input
            
            flag = 0
            enter_time = time.time()
            img_id = 0
            for origin_input in test_data_loader:
                ## origin_input: list of bs * (input_discs, label)
                batch_input_dicts = []
                batch_label = []
                for bs_id in range(len(origin_input)):
                    # Input Dict
                    batch_input_dicts.append(origin_input[bs_id][0])
                    # Label
                    if type(origin_input[bs_id][1]) != str:
                        batch_label.append(origin_input[bs_id][1].tolist())
                    else:
                        batch_label.append(origin_input[bs_id][1])

                bs = len(origin_input)
                seq_length = batch_input_dicts[0]['input_ids'].shape[-1]

                data_inputs = {}
                for key_name in batch_input_dicts[0].keys():
                    if isinstance(batch_input_dicts[0][key_name], torch.Tensor):
                        data_inputs[key_name] = torch.stack( [batch_input_dicts[i][key_name] for i in range(len(batch_input_dicts))] )
                    else:
                        data_inputs[key_name] = [batch_input_dicts[i][key_name] for i in range(len(batch_input_dicts))]         

                # print('DLG batch_input_dicts:',len(batch_input_dicts),batch_input_dicts[0].keys())

                # origin data
                sample_origin_input_ids = data_inputs['input_ids']
                # print('sample_origin_input_ids:',sample_origin_input_ids.shape) # 2, 160 bs, seq_len

                sample_origin_image_list = data_inputs['pixel_values']
                sample_origin_image_list = [ sample_origin_image[0]
                    for sample_origin_image in sample_origin_image_list]
                # list of bs * [image1, image2...]
                # print('sample_origin_image_list:',len(sample_origin_image_list),sample_origin_image_list[0].shape)


                sample_origin_image_bound = data_inputs['image_bound']# bs, 1, 2
                # print('sample_origin_image_bound:',sample_origin_image_bound)

                # train batch (without update)
                self.top_vfl.train_batch([[batch_input_dicts, batch_label]], batch_label)

                # real grad
                original_dy_dx = self.top_vfl.parties[1].weights_grad_a # gradient calculated for local model update
                print(f'original_dy_dx:{type(original_dy_dx)} {len(original_dy_dx)}')


                ####### begin attack
                # initial guess
                dummy_label = batch_label # list of bs * list of label ids

                if 'attention_mask' in data_inputs.keys() and data_inputs['attention_mask'] != None:
                    dummy_attention_mask = data_inputs['attention_mask'].to(self.device)
                else:
                    dummy_attention_mask = None
                
                dummy_vllm_embedding = ( 
                    self.top_vfl.parties[0].local_model.llm.model.embed_tokens(sample_origin_input_ids) * self.top_vfl.parties[0].local_model.llm.config.scale_emb
                )
                dummy_vllm_embedding.requires_grad = False
                print('dummy_vllm_embedding:',dummy_vllm_embedding[0,:5])
                # torch.zeros([bs,seq_length,self.args.model_embedded_dim]).type(torch.float32).to(self.device)
                
                dummy_image_list = copy.deepcopy(sample_origin_image_list)
                # dummy_image_list = [ torch.rand(sample_origin_image.size()).type(torch.float32).to(self.device)
                #     for sample_origin_image in sample_origin_image_list]
                for dummy_img in dummy_image_list:
                    dummy_img[0].requires_grad = True
                # print('dummy_image_list:',len(dummy_image_list),dummy_image_list[0].shape)

                optimizer = torch.optim.Adam(dummy_image_list, lr=self.lr)
                
                def get_cost(dummy_embedding):
                    # compute dummy result
                    embedding_dict = {"pixel_values":[dummy_image_list], # bs, img_size, embed_dim2304 [image]
                        "vllm_embedding":dummy_vllm_embedding, # bs, seq_len, embed_dim2304 [text]
                        "image_bound":sample_origin_image_bound} # bs, 1, 2
                    dummy_embedding = self.top_vfl.parties[0].local_model.get_vllm_embedding(
                        data = embedding_dict 
                    )[0] # 1, 160, 2304

                    dummy_input = {
                    'input_ids':None, 'attention_mask':dummy_attention_mask,\
                    'inputs_embeds':dummy_embedding
                    }

                    dummy_intermediate_dict = self.top_vfl.parties[0].local_model(**dummy_input)
                    self.top_vfl.parties[0].local_model._clear_past_key_values()
                    
                    dummy_final_pred = self.top_vfl.parties[1].global_model(**dummy_intermediate_dict)

                    dummy_loss = self.cal_loss(dummy_final_pred, dummy_label)
                
                    dummy_dy_dx_a = torch.autograd.grad(dummy_loss, active_model_body_params, create_graph=True, allow_unused=True)
                    # print(f'dummy_dy_dx_a:{type(dummy_dy_dx_a)} {len(dummy_dy_dx_a)}')

                    # loss: L-L'
                    grad_diff = 0
                    for (gx, gy) in zip(dummy_dy_dx_a, original_dy_dx):
                        if gx != None:
                            grad_diff += ((gx - gy) ** 2).sum()

                    return grad_diff

                cost_function = torch.tensor(10000000)
                last_cost = torch.tensor(10000000)
                _iter = 0
                while _iter<self.epochs: # cost_function.item()>=0.1 and 
                    optimizer.zero_grad()
                    cost_function = get_cost(dummy_image_list)
                    
                    cost_function.backward()
                    optimizer.step()
                    _iter+=1
                    # if _iter%20 == 0:
                    print('=== iter ',_iter,'  cost:',cost_function.item())
                    
                

                # recover images from dummy embeddings
                img_criterion = nn.MSELoss()
                
                for _i in range(len(sample_origin_image_list)):
                    img_mse = img_criterion(sample_origin_image_list[_i],dummy_image_list[_i]).item()
                    
                    rand_image = torch.randn(sample_origin_image_list[_i].size()).to(self.device)
                    rand_mse = img_criterion(sample_origin_image_list[_i], rand_image).item()
                    
                    def save_tensor_to_image(input_tensor,img_dir ,image_name):
                        img = reverse_vis_processor(input_tensor)
                        img.save(img_dir+f'{image_name}.jpg')
                        print('save:',img_dir+f'{image_name}.jpg')
                    img_dir = f"{self.args.exp_res_dir}/{self.attack_name}/"
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    save_tensor_to_image(dummy_image_list[0],img_dir, f'dummy_{img_id}')
                    save_tensor_to_image(sample_origin_image_list[0],img_dir,f'origin_{img_id}')
                    img_id = img_id + 1

                    # print('img_mse:',img_mse,'  rand_mse:',rand_mse)

                    attack_result.loc[len(attack_result)] = [sample_origin_image_list[_i].shape,img_mse,rand_mse ]

                    # assert 1>2
                
                del(dummy_vllm_embedding)
                del(dummy_image_list)
                del(dummy_attention_mask)
                    
                

            end_time = time.time()
        
        attack_total_time = end_time - enter_time
        Image_Mse = attack_result['Img_Mse'].mean()
        Rand_Image_Mse = attack_result['Rand_Img_Mse'].mean()


        print('FINAL: Image_Mse:',Image_Mse,'  Rand_Image_Mse:',Rand_Image_Mse)

        
        

        # model_name = self.args.model_list['0']['type']
        # if self.args.pretrained == 1:
        #     result_path = f'./exp_result/{str(self.args.dataset)}/{self.attack_name}/{self.args.defense_name}_{self.args.defense_param}_pretrained_{str(model_name)}/'
        # else:
        #     result_path = f'./exp_result/{str(self.args.dataset)}/{self.attack_name}/{self.args.defense_name}_{self.args.defense_param}_finetuned_{str(model_name)}/'

        # if not os.path.exists(result_path):
        #     os.makedirs(result_path)
        # result_file_name = result_path + f'{self.args.pad_info}_{str(Precision)}_{str(Recall)}.csv'
        # print(result_file_name)
        # attack_result.to_csv(result_file_name)

        
        return Image_Mse, Rand_Image_Mse, attack_total_time
        # else:
        #     return {
        #         'precision': Precision,
        #         'recall': Recall,
        #         'img_mse': Image_Mse,
        #         'rand_img_mse': Rand_Image_Mse,
        #         'attack_total_time': attack_total_time,
        #     }