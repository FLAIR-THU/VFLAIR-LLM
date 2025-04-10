import json
import sys, os

sys.path.append(os.pardir)
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from loguru import logger
from party.party import Party
from party.llm_party import Party as Party_LLM
from utils.basic_functions import cross_entropy_for_onehot, tf_distance_cov_cor, pairwise_dist
from utils import timer
from dataset.party_dataset import ActiveDataset
from framework.client.DistributedCommunication import convert_pred_to_msg, convert_msg_to_pred, convert_tensor_to_batch_msg, convert_msg_to_tensor, convert_to_msg
from models.llm_models.base import VFLModelIntermediate
from config import vfl_basic_config
import copy


class ActiveParty_LLM(Party_LLM):
    def __init__(self, args, index, need_data=True, need_model=True):
        print(f'###### initialize ActiveParty_LLM : Party {index} ######')
        logger.debug(f'running on cuda{os.getenv("CUDA_VISIBLE_DEVICES").split(",")[torch.cuda.current_device()]}')

        super().__init__(args, index, need_data=need_data, need_model=need_model)
        self.name = "server#" + str(index + 1)
        self.criterion = cross_entropy_for_onehot
        # self.encoder = args.encoder

        self.train_index = None  # args.idx_train
        self.test_index = None  # args.idx_test

        self.gt_one_hot_label = None
        
        # store attributes of model slices
        self.input_tensors = [{} for i in range(args.k-1)]  # client_number * input intermediate type:dict[int,torch.Tensor]
        self.input_attention_mask = [{} for i in range(args.k-1)]  # client_number * input attention mask type:dict[int,torch.Tensor]
        self.output_tensors = [{} for i in range(args.k-1)]  # client_number * output embeddings type:dict[int,torch.Tensor]
        self.output_attention_mask = [{} for i in range(args.k-1)]  # client_number * output attention mask type:dict[int,torch.Tensor]

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])

        self.global_output = None 
        self.global_output_dict = {} 
        
        self.global_loss = None
        self.global_loss_dict = {}
        
        self.global_gradient = None
        self.global_gradient_dict = {}
        
        self.weights_grad_a = None
        self.weights_grad_a_list = [ None for _i in range(self.args.k-1)]

        self.encoder_hidden_states = None
        self.encoder_attention_mask = None
        self.first_epoch_state = None

    # def prepare_data_loader(self, **kwargs):
    #     super().prepare_data_loader(self.args.batch_size, self.args.need_auxiliary)

    def get_output_tensors(self):
        return convert_to_msg(self.output_tensors)

    def get_output_attention_mask(self):
        return convert_to_msg(self.output_attention_mask)

    def get_global_gradient(self):
        return convert_to_msg(self.global_gradient)

    def get_weights_grad_a(self):
        return convert_to_msg(self.weights_grad_a)

    def save_model_body(self):
        self.first_epoch_state = {
            "active_model_body": copy.deepcopy(self.global_model).to("cpu") if self.global_model != None else None,
        }

    def get_global_model(self):
        return copy.deepcopy(self.global_model).to("cpu") if self.global_model != None else None,

    def get_global_parameters(self):
        return self.first_epoch_state['active_model_body'].parameters()

    def model_body_forward(self, intermediate):
        resp = self.first_epoch_state['active_model_body'](**intermediate)
        return resp

    def prepare_data(self, args, index):
        print('Active Party has no data, only global model')

    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def receive_attention_mask(self, attention_mask):
        self.local_batch_attention_mask = attention_mask

    def receive_token_type_ids(self, token_type_ids):
        self.local_batch_token_type_ids = token_type_ids

    @timer()
    def forward(self, model_index, client_id, **kwargs):
        # logger.debug(f"model_{model_index} forward")

        self.input_tensors[client_id][model_index] = kwargs.get('inputs_embeds')
        self.input_attention_mask[client_id][model_index] = kwargs.get('attention_mask')
        
        self._tensor_to_device(kwargs , self.models[model_index].device)
        
        # print(f"model_{model_index} forward:{kwargs.keys()}")
        resp = self.models[model_index](**kwargs)

        if model_index == self.args.vfl_model_slice_num - 1:
            if not self.args.task_type == 'QuestionAnswering':
                self.output_tensors[client_id][model_index] = resp.get('logits')
            else:
                self.output_tensors[client_id][model_index] = resp.get('start_logits') + resp.get('end_logits') 
            self.output_attention_mask[client_id][model_index] = None
        else:
            if resp.get('inputs_embeds') != None:
                self.output_tensors[client_id][model_index] = resp.get('inputs_embeds')
            else: # for encoder-decoder inference
                self.output_tensors[client_id][model_index] = resp.get('encoder_outputs')['last_hidden_state']
            self.output_attention_mask[client_id][model_index] = resp.get('attention_mask')

        return resp #self._detach_tensor(resp)


    def _do_aggregate_remote(self, pred_list):
        new_dict = convert_msg_to_pred(pred_list)
        if self.args.model_type == 'XLNet':
            new_dict['output_g'] = None
        result = self.aggregate([new_dict])

        if not self.args.task_type or self.args.task_type == 'CausalLM': 
            if 'logits' in result:
                return convert_tensor_to_batch_msg(result.logits, 'test_logit')
            else:
                return convert_pred_to_msg(result, 'test_logit')
        elif self.args.task_type == 'SequenceClassification':  
            return convert_pred_to_msg(result, 'test_logit')
        
        # elif self.args.task_type == 'QuestionAnswering': 
        #     return {
        #         "requires_grad": True,
        #         "start_logits": result.start_logits.tolist(),
        #         "end_logits": result.end_logits.tolist(),
        #     }
        elif self.args.task_type == 'QuestionAnswering':  # self.passive_pred_list[0] = [intermediate, attention_mask]
            return convert_pred_to_msg(result, 'test_logit')
        elif self.args.task_type == 'DevLLMInference':
            return convert_pred_to_msg(result)
        else:
            assert 1 > 2, 'Task type no supported'

    def aggregate_remote(self, pred_list):
        return self._do_aggregate_remote(pred_list)

    @timer()
    def aggregate(self, passive_pred, current_client_id, use_cache=False, test=False):
        self._tensor_to_device(passive_pred, self.device)
        intermediate = self.forward(model_index=1, client_id = current_client_id, **passive_pred)  # use_cache = use_cache,return_dict=True
        self.global_output_dict[current_client_id] = intermediate
        return self._detach_tensor(self.global_output_dict[current_client_id])


    def receive_loss_and_gradients_remote(self, data, client_id):
        gradients = convert_msg_to_tensor(data)
        gradients = gradients.to(self.device)
        self.receive_loss_and_gradients(gradients,client_id)

    def receive_loss_and_gradients(self, gradients, client_id):
        self.global_gradient_dict[client_id] = gradients

    def aggregate_gradients(self):
        num_tensors = len(self.global_gradient_dict)
        
        # Calculate the total sum of tensors
        total_sum = torch.zeros_like(list(self.global_gradient_dict.values())[0])
        for tensor in self.global_gradient_dict.values():
            total_sum += tensor

        # Calculate the average tensor
        average_tensor = total_sum / num_tensors

        self.global_gradient = average_tensor
        
        
    def global_LR_decay(self, i_epoch):
        if self.global_model_optimizer != None:
            eta_0 = self.args.main_lr
            eta_t = eta_0 / (np.sqrt(int(i_epoch) + 1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t
        elif self.lr_schedulers.get(1):
            self.lr_schedulers[1].step()

    def cal_passive_local_gradient(self, ik, remote=True):
        if remote:
            ik = int(ik)
        if self.args.vfl_model_slice_num== 2 and self.args.model_architect == 'TQA':
            global_output = self.global_output_dict[ik]
            logits = torch.cat((global_output.start_logits.unsqueeze(-1), global_output.end_logits.unsqueeze(-1)), dim=-1)
            passive_local_gradient = torch.autograd.grad(logits, self.input_tensors[ik][1],\
                grad_outputs=self.global_gradient, retain_graph=True)[0].detach().clone()
        else:
            _device = self.output_tensors[ik][1].device
            passive_local_gradient = torch.autograd.grad(self.output_tensors[ik][1], self.input_tensors[ik][1].to(_device), \
                                    grad_outputs=self.global_gradient.to(_device), retain_graph=True)[0].detach().clone()
        if remote:
            return convert_tensor_to_batch_msg(passive_local_gradient, 'test_logit')
        return passive_local_gradient
    
    def global_backward(self):
        '''
        3-slice: model body backward
        2-slive: model tail backward
        '''
        def cal_avg_grad(weights_grad_a_list):
            '''
            input:  weights_grad_a_list: client_number * [grad_tensor1, grad_tensor2...]
            output: avg_weights_grad_a: [avg_grad_tensor1, avg_grad_tensor2...]
            '''
            avg_weights_grad_a = []
            for _i in range( len(weights_grad_a_list[0]) ):
                if weights_grad_a_list[0][_i] == None:
                    avg_weights_grad_a.append(None)
                else:
                    grad_list = [sublist[_i] for sublist in weights_grad_a_list]
                    total_sum = torch.zeros_like(grad_list[0])
                    for tensor in grad_list:
                        total_sum += tensor
                    avg_grad = total_sum / len(grad_list)
                    avg_weights_grad_a.append(avg_grad)
            return avg_weights_grad_a
                
        if self.global_model_optimizer != None:
            # trainable layer parameters
            global_model_params = []
            # for param in self.models[1].parameters():
            #     if param.requires_grad:
            #         global_model_params.append(param)

            global_model_params_name = []
            for name,param in self.models[1].named_parameters():
                if param.requires_grad:
                    global_model_params.append(param)
                    global_model_params_name.append(name)

            if self.args.vfl_model_slice_num==2 and self.args.model_architect == 'TQA':
                # update global model
                self.global_model_optimizer.zero_grad()

                self.weights_grad_a_list = []
                for client_id in range(self.args.k-1):
                    # load grads into parameters
                    logits = torch.cat((self.global_output_dict[client_id].start_logits.unsqueeze(-1), self.global_output_dict[client_id].end_logits.unsqueeze(-1)), dim=-1)
                    client_global_gradient = self.global_gradient_dict[client_id].to(self.output_tensors[client_id][1].device)
                    weights_grad_a = torch.autograd.grad(logits,
                                                        global_model_params, 
                                                        grad_outputs=client_global_gradient, 
                                                        allow_unused=True,
                                                        retain_graph=True)
                    self.weights_grad_a_list.append( list(weights_grad_a) )
                    # weights_grad_a_start = torch.autograd.grad(self.global_output_dict[client_id].start_logits,
                    #                                         global_model_params, 
                    #                                         grad_outputs=self.global_gradient_dict[client_id], 
                    #                                         retain_graph=True,allow_unused=True)
                    # weights_grad_a_end = torch.autograd.grad(self.global_output_dict[client_id].end_logits,
                    #                                         global_model_params, 
                    #                                         grad_outputs=self.global_gradient_dict[client_id], 
                    #                                         retain_graph=True,allow_unused=True)

                    # weights_grad_a = []
                    # for _i in range(len(weights_grad_a_start)):
                    #     weights_grad_a.append(weights_grad_a_start[_i] + weights_grad_a_end[_i] if weights_grad_a_start[_i]!= None else None)
                    # self.weights_grad_a_list.append(weights_grad_a)
                    
                    # self.weights_grad_a_list.append( list(weights_grad_a) )
            else:
                self.global_model_optimizer.zero_grad()
                
                self.weights_grad_a_list = []
                for client_id in range(self.args.k-1):
                    client_global_gradient = self.global_gradient_dict[client_id].to(self.output_tensors[client_id][1].device)
                    # print(f'client_id={client_id} self.output_tensors[client_id][1]:',type(self.output_tensors[client_id][1]))
                    # print('global_model_params:',len(global_model_params))
                    # print('client_global_gradient:',type(client_global_gradient),len(client_global_gradient))
                    weights_grad_a = torch.autograd.grad(self.output_tensors[client_id][1],
                                                            global_model_params, 
                                                            grad_outputs=client_global_gradient, 
                                                            allow_unused=True,
                                                            retain_graph=True)
                    self.weights_grad_a_list.append( list(weights_grad_a) )
            self.weights_grad_a = tuple( cal_avg_grad(self.weights_grad_a_list) )
            
            for w, g in zip(global_model_params, self.weights_grad_a):
                if w.requires_grad and g!=None:
                    if w.grad != None:
                        w.grad += g.detach()
                    else:
                        w.grad = g.detach()

            self.global_model_optimizer.step()

    @property
    def device(self):
        return self.models[1].device

class ActiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)
        self.criterion = cross_entropy_for_onehot
        self.encoder = args.encoder
        # print(f"in active party, encoder=None? {self.encoder==None}, {self.encoder}")
        self.train_index = args.idx_train
        self.test_index = args.idx_test

        self.gt_one_hot_label = None

        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])

        self.global_pred = None
        self.global_loss = None

    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = ActiveDataset(self.train_data, self.train_label)
        self.test_dst = ActiveDataset(self.test_data, self.test_label)
        if self.args.need_auxiliary == 1:
            self.aux_dst = ActiveDataset(self.aux_data, self.aux_label)
            # self.aux_loader = DataLoader(self.aux_dst, batch_size=batch_size,shuffle=True)

    def update_local_pred(self, pred):
        self.pred_received[self.args.k - 1] = pred

    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def aggregate(self, pred_list, gt_one_hot_label, test=False):
        if self.args.dataset == 'cora' and self.args.apply_trainable_layer == 1:
            pred = self.global_model(pred_list, self.local_batch_data)
        else:
            pred = self.global_model(pred_list)

        if self.train_index != None:  # for graph data
            if test == False:
                loss = self.criterion(pred[self.train_index], gt_one_hot_label[self.train_index])
            else:
                loss = self.criterion(pred[self.test_index], gt_one_hot_label[self.test_index])
        else:
            loss = self.criterion(pred, gt_one_hot_label)

        # ########## for active mid model loss (start) ##########
        if self.args.apply_mid == True and (self.index in self.args.defense_configs['party']):
            # print(f"in active party mid, label={gt_one_hot_label}, global_model.mid_loss_list={self.global_model.mid_loss_list}")
            assert len(pred_list) - 1 == len(self.global_model.mid_loss_list)
            for mid_loss in self.global_model.mid_loss_list:
                loss = loss + mid_loss
            self.global_model.mid_loss_list = [torch.empty((1, 1)).to(self.args.device) for _ in
                                               range(len(self.global_model.mid_loss_list))]
        # ########## for active mid model loss (end) ##########
        elif self.args.apply_dcor == True and (self.index in self.args.defense_configs['party']):
            # print('dcor active defense')
            self.distance_correlation_lambda = self.args.defense_configs['lambda']
            # loss = criterion(pred, gt_one_hot_label) + self.distance_correlation_lambda * torch.mean(torch.cdist(pred_a, gt_one_hot_label, p=2))
            for ik in range(self.args.k - 1):
                loss += self.distance_correlation_lambda * torch.log(
                    tf_distance_cov_cor(pred_list[ik], gt_one_hot_label))  # passive party's loss
        return pred, loss

    def gradient_calculation(self, pred_list, loss):
        pred_gradients_list = []
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients_list.append(torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True))
            # print(f"in gradient_calculation, party#{ik}, loss={loss}, pred_gradeints={pred_gradients_list[-1]}")
            pred_gradients_list_clone.append(pred_gradients_list[ik][0].detach().clone())
        # self.global_backward(pred, loss)
        return pred_gradients_list, pred_gradients_list_clone

    def give_gradient(self):
        pred_list = self.pred_received

        if self.gt_one_hot_label == None:
            print('give gradient:self.gt_one_hot_label == None')
            assert 1 > 2

        self.global_pred, self.global_loss = self.aggregate(pred_list, self.gt_one_hot_label)
        pred_gradients_list, pred_gradients_list_clone = self.gradient_calculation(pred_list, self.global_loss)
        # self.local_gradient = pred_gradients_list_clone[self.args.k-1] # update local gradient

        if self.args.defense_name == "GradPerturb":
            self.calculate_gradient_each_class(self.global_pred, pred_list)

        return pred_gradients_list_clone

    def update_local_gradient(self, gradient):
        self.local_gradient = gradient

    def global_LR_decay(self, i_epoch):
        if self.global_model_optimizer != None:
            eta_0 = self.args.main_lr
            eta_t = eta_0 / (np.sqrt(i_epoch + 1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t

    def global_backward(self):

        if self.global_model_optimizer != None:
            # active party with trainable global layer
            _gradients = torch.autograd.grad(self.global_loss, self.global_pred, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()

            # update global model
            self.global_model_optimizer.zero_grad()
            parameters = []
            if (self.args.apply_mid == True) and (self.index in self.args.defense_configs['party']):
                # mid parameters
                for mid_model in self.global_model.mid_model_list:
                    parameters += list(mid_model.parameters())
                # trainable layer parameters
                if self.args.apply_trainable_layer == True:
                    parameters += list(self.global_model.global_model.parameters())

                # load grads into parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, parameters, grad_outputs=_gradients_clone,
                                                     retain_graph=True)
                for w, g in zip(parameters, weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()

            else:
                # trainable layer parameters
                if self.args.apply_trainable_layer == True:
                    # load grads into parameters
                    weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.parameters(),
                                                         grad_outputs=_gradients_clone, retain_graph=True)
                    for w, g in zip(self.global_model.parameters(), weights_grad_a):
                        if w.requires_grad:
                            w.grad = g.detach()
                # non-trainabel layer: no need to update
            self.global_model_optimizer.step()

    def calculate_gradient_each_class(self, global_pred, local_pred_list, test=False):
        # print(f"global_pred.shape={global_pred.size()}") # (batch_size, num_classes)
        self.gradient_each_class = [[] for _ in range(global_pred.size(1))]
        one_hot_label = torch.zeros(global_pred.size()).to(global_pred.device)
        for ic in range(global_pred.size(1)):
            one_hot_label *= 0.0
            one_hot_label[:, ic] += 1.0
            if self.train_index != None:  # for graph data
                if test == False:
                    loss = self.criterion(global_pred[self.train_index], one_hot_label[self.train_index])
                else:
                    loss = self.criterion(global_pred[self.test_index], one_hot_label[self.test_index])
            else:
                loss = self.criterion(global_pred, one_hot_label)
            for ik in range(self.args.k):
                self.gradient_each_class[ic].append(
                    torch.autograd.grad(loss, local_pred_list[ik], retain_graph=True, create_graph=True))
        # end of calculate_gradient_each_class, return nothing
