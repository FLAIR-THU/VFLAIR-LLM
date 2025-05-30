import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import string
from utils.squad_utils import normalize_answer
from random import randrange
from typing import List, Dict, Optional, Sequence, Union
import copy
from PIL import Image
from torchvision import transforms
# import logging
# logger = logging.getLogger(__name__)

# from minigpt4.caption_datasets import CaptionDataset

class SimpleDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i, target_i = self.data[item_idx], self.labels[item_idx]
        return torch.tensor(data_i.clone().detach(), dtype=torch.float32), torch.tensor(target_i.clone().detach(),
                                                                                        dtype=torch.long)


class PassiveDataset_LLM(Dataset):
    def __init__(self, args, texts: Union[np.array, List[Dict]] ,labels, split_name='test'):
        '''
        texts: np.array
        '''
        self.args = args
        self.labels = []
        self.features = []
        self.input_dicts = []

        if args.task_type == 'SequenceClassification':
            texts = np.array(texts)
            for _text in texts:
                if len(texts.shape) == 1:  # input: single sentence
                    if args.padding != "do_not_pad" and args.padding_type == "inside":  # [PAD] between [CLS][SEP]
                        text_tokens = args.tokenizer.tokenize(_text)

                        pad_length = max(args.max_length - len(text_tokens), 0)
                        for _pad in range(pad_length):
                            if args.padding_side == 'right':
                                text_tokens.append(args.tokenizer.pad_token)
                            elif args.padding_side == 'left':
                                text_tokens.insert(0, args.tokenizer.pad_token)
                            elif args.padding_side == 'random':
                                text_tokens.insert(randrange(len(text_tokens) + 1), args.tokenizer.pad_token)

                        _text = " ".join(text_tokens)
                        # print('after pad:', _text)

                        ids = args.tokenizer(_text, truncation=args.truncation, max_length=args.max_length, \
                                             return_tensors='pt', add_special_tokens=args.add_special_tokens)

                        # for _pos in range(ids['attention_mask'].shape[1]):
                        #     if ids['input_ids'][0][_pos] == args.tokenizer.pad_token_id:
                        #         ids['attention_mask'][0][_pos] = 0
                    else:  # [PAD] outside [CLS][SEP]
                        ids = args.tokenizer(_text, \
                                             padding=args.padding, truncation=args.truncation, \
                                             max_length=args.max_length, return_tensors='pt',
                                             add_special_tokens=args.add_special_tokens)
                elif len(texts.shape) > 1:  # input: sentence pairs
                    try:
                        ids = args.tokenizer(_text[0], _text[1], \
                                             padding=args.padding, truncation=args.truncation, \
                                             max_length=args.max_length, return_tensors='pt')
                    except:
                        assert 1 > 2
                else:
                    print(texts.shape)
                    assert 1 > 2, 'text input shape not supported'

                self.input_dicts.append(ids)

            print(type(labels[:2]), labels[:2])
            if self.args.num_classes == 1:
                self.labels = torch.tensor(labels, dtype=torch.float32)
            else:
                self.labels = torch.tensor(labels)

        elif args.task_type == 'CausalLM':
            if isinstance(texts[0], Dict):
                self.input_dicts = texts
                self.labels = labels
                return
            if split_name == 'test':
                if isinstance(texts[0],Dict):
                    self.input_dicts=texts
                    self.labels=labels
                else:
                    for i in range(len(texts)):
                        ids = args.tokenizer(texts[i], \
                        padding=args.padding,truncation=args.truncation ,\
                        max_length=args.max_length,return_tensors='pt')

                        # if i == 0:
                        #     print('TEXT:',texts[i])
                        #     print('text_id:',ids['input_ids'].shape, ids['input_ids'])
                        #     print('label:',labels[i],args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                        #     print('-'*25)

                        self.labels.append( args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                        self.input_dicts.append(ids)
            else:
                for i in range(len(texts)):
                    ids = args.tokenizer(texts[i], \
                    padding=args.padding,truncation=args.truncation ,\
                    max_length=args.max_length,return_tensors='pt')

                    # if i == 0:
                    #     print('TEXT:',texts[i])
                    #     print('text_id:',ids['input_ids'].shape, ids['input_ids'])
                    #     print('label:',labels[i],args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                    #     print('-'*25)

                    self.labels.append(ids['input_ids'])#args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                    self.input_dicts.append(ids)
        
        elif args.task_type == 'QuestionAnswering':
            # labels : bs * [start_position, end_position]
            # texts: bs * [feature]
            for i in range(len(texts)):
                _feature = texts[i]
                inputs = {
                    'input_ids': _feature["input_ids"],
                    'attention_mask': _feature["input_mask"],
                    'token_type_ids': _feature["segment_ids"],

                    'feature': {
                        'token_to_orig_map': _feature["token_to_orig_map"],
                        'token_is_max_context': _feature["token_is_max_context"],
                        'len_tokens': len(_feature["tokens"])
                    }
                }
                self.input_dicts.append(inputs)
                self.labels.append( labels[i] )
                # test: [ [start_position, end_position] ]
                # train: [ [ [start_position1,start_position2,start_position3],[end_position1,end_position2,end_position3] ] ]

            print(f'---- {split_name} -----')
            print('self.labels:',self.labels[:3])
            # self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance( input_dict[key_name] , dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        if self.args.model_type == 'T5':
            if self.args.tokenizer.bos_token_id == None:
                self.args.tokenizer.bos_token_id = 101 # set to 101 defaultly
            
            if self.split_name == 'train':
                input_dict['decoder_input_ids'] = input_dict['input_ids']
            else:
                bos_tensor = torch.tensor( [self.args.tokenizer.bos_token_id] ).to(input_dict['input_ids'].device)
                # input_dict['decoder_input_ids'] = torch.cat((bos_tensor, input_dict['input_ids'][1:]), dim=0)
                input_dict['decoder_input_ids'] = bos_tensor

        
        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label

class TextVQADataset_forMiniCPM(Dataset):
    def __init__(self, args, data , labels, transform, split_name='test'):
        '''
        data_dict = {
                'question':data[idx]['question'],
                'img_path':os.path.join(image_dir, f"{data[idx]['image_id']}.jpg")
            }
        '''
        self.args = args
        self.split_name = split_name
        self.data = data
        self.labels = labels
        self.tokenizer = args.tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.split_name == 'test':
            prompt = "Answer the question directly with single short sentence." + '\n'
        
            data_input = self.data[idx]
            answer = self.labels[idx]

            input_dict = {
                    'question':prompt + data_input['question'],
                    'image':Image.open(data_input['img_path']).convert('RGB')
            }
        
            bos_tensor = torch.tensor( [self.args.tokenizer.bos_token_id] ).to(input_dict['input_ids'].device)
            input_dict['decoder_input_ids'] = bos_tensor
        
        else:
            data_input = self.data[idx]
            answer = self.labels[idx][0]

            images_dict = { "<image>" : Image.open(data_input['img_path']).convert('RGB') }

            conversation = [{'role': 'user', 'content': data_input['question']},
                    {'role': 'assistant', 'content': answer}]

            ret = preprocess(
                images_dict,
                conversation,
                self.tokenizer,
                self.transform,
                # query_nums=self.query_nums,
                # slice_config=self.slice_config,
                # llm_type=self.llm_type,
                # patch_size=self.patch_size,
                # batch_vision=self.batch_vision,
                max_length=self.args.max_length
            )


            padded_input_ids = torch.cat((ret["input_ids"], torch.full((self.args.max_length - len(ret["input_ids"]),), 
                fill_value=self.tokenizer.pad_token_id)))

            attention_mask = torch.cat((ret["input_ids"], torch.full((self.args.max_length - len(ret["input_ids"]),), 
                fill_value=0)))
            # torch.tensor([1 if token != self.tokenizer.pad_token_id else 0 for token in padded_input_ids])

            position_ids = torch.cat((ret["position_ids"], torch.full((self.args.max_length - len(ret["input_ids"]),), 
                fill_value=0)))

            input_dict = dict(
                input_ids= padded_input_ids.to(self.args.device),
                position_ids= position_ids.to(self.args.device),
                attention_mask= attention_mask.to(self.args.device),
                pixel_values=[pixel_value.to(self.args.device) for pixel_value in ret["pixel_values"]],
                tgt_sizes=ret["tgt_sizes"],
                image_bound=ret["image_bound"],
            )
            input_dict['decoder_input_ids'] = input_dict['input_ids']
            label = torch.cat((ret["target"], torch.full((self.args.max_length - len(ret["target"]),),
            fill_value=-100)))
            return input_dict, label

class TextVQADataset_forBlip(Dataset):
    def __init__(self, args, data , labels, split_name='test'):
        self.args = args
        self.split_name = split_name
        self.data = data
        self.labels = labels
        self.tokenizer = args.tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.split_name == 'test':
            prompt = "Answer the question directly with single short sentence." + "\n<Image><ImageHere></Image>\n "
        
            data_input = self.data[idx]
            answer = self.labels[idx]

            input_dict = {
                    'prompt':prompt + data_input['question'],
                    'images': Image.open(data_input['img_path']).convert('RGB')
            }
        
            bos_tensor = torch.tensor( [self.args.tokenizer.bos_token_id] ).to(self.args.device)
            input_dict['decoder_input_ids'] = bos_tensor
            
            return input_dict, answer
        else:
            data_input = self.data[idx]
            answer = self.labels[idx][0]

            images_dict = { "<image>" : Image.open(data_input['img_path']).convert('RGB') }

            conversation = [{'role': 'user', 'content': data_input['question']},
                    {'role': 'assistant', 'content': answer}]

            ret = preprocess(
                images_dict,
                conversation,
                self.tokenizer,
                self.transform,
                # query_nums=self.query_nums,
                # slice_config=self.slice_config,
                # llm_type=self.llm_type,
                # patch_size=self.patch_size,
                # batch_vision=self.batch_vision,
                max_length=self.args.max_length,
            )


            padded_input_ids = torch.cat((ret["input_ids"], torch.full((self.args.max_length - len(ret["input_ids"]),), 
                fill_value=self.tokenizer.pad_token_id)))

            attention_mask = torch.cat((ret["input_ids"], torch.full((self.args.max_length - len(ret["input_ids"]),), 
                fill_value=0)))
            # torch.tensor([1 if token != self.tokenizer.pad_token_id else 0 for token in padded_input_ids])

            position_ids = torch.cat((ret["position_ids"], torch.full((self.args.max_length - len(ret["input_ids"]),), 
                fill_value=0)))

            input_dict = dict(
                input_ids= padded_input_ids.to(self.args.device),
                position_ids= position_ids.to(self.args.device),
                attention_mask= attention_mask.to(self.args.device),
                pixel_values=[pixel_value.to(self.args.device) for pixel_value in ret["pixel_values"]],
                tgt_sizes=ret["tgt_sizes"],
                image_bound=ret["image_bound"],
            )
            input_dict['decoder_input_ids'] = input_dict['input_ids']
            label = torch.cat((ret["target"], torch.full((self.args.max_length - len(ret["target"]),),
            fill_value=-100)))
            return input_dict, label



def preprocess(
    images_dict,
    conversations,
    tokenizer,
    transform,
    query_nums=64,
    slice_config=None,
    llm_type=None,
    patch_size=14,
    batch_vision=False,
    max_length=2048,
):
    """
    single(multi) image(s) preprocess, the image(s) will be placed at the top of the conversation
    """
    conversations = copy.deepcopy(conversations)
    assert len(conversations) > 1, "conversations length must large than 2"
    assert conversations[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    default_image_placeholder = (
        tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    )
    new_schema = False
    use_image_id = False
    if llm_type=='qwen2':
        new_schema = True
        use_image_id = True
    image_placeholder_dict = {}
    images = []
    image_id_cnt = 0 
    for img_name, image in images_dict.items():
        if slice_config:
            source_image, patches, best_grid = slice_image(
                image,
                slice_config["max_slice_nums"],
                slice_config["scale_resolution"],
                slice_config["patch_size"],
            )
            images.append(source_image)
            image_placeholder = default_image_placeholder
            if len(patches) > 0:
                for i in range(len(patches)):
                    for j in range(len(patches[0])):
                        images.append(patches[i][j])
                if use_image_id:
                    image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                    image_id_cnt += 1
                image_placeholder += get_grid_placeholder(
                    tokenizer, best_grid, query_nums, new_schema = new_schema)
            image_placeholder_dict[img_name] = image_placeholder
        else:
            images.append(image)
            if use_image_id:
                image_placeholder = f'{tokenizer.im_id_start}{image_id_cnt}{tokenizer.im_id_end}' + image_placeholder
                image_id_cnt += 1
            else:
                image_placeholder = default_image_placeholder
            image_placeholder_dict[img_name] = image_placeholder
    
    images = [transform(i) for i in images]
    
    if len(images_dict) == 1 and "<image>" in images_dict:       
        if "<image>" in conversations[0]["content"]:
            conversations[0]["content"] = conversations[0]["content"].replace(
                "<image>", image_placeholder
            )
        else:
            conversations[0]["content"] = (
                image_placeholder + "\n" + conversations[0]["content"]
            )
        input_dict = conversation_to_ids(conversations, tokenizer, llm_type, new_schema, max_length)
    else:
        pattern = r'<image_\d+>'
        new_conversations = []
        for conversation in conversations:
            content = conversation['content']
            parts = re.split(f'({pattern})', content)
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                if re.match(pattern, part):  
                    if part in image_placeholder_dict:
                        parts[i] = image_placeholder_dict[part] 
                    else:
                        raise Exception(f"not found {part} in image dict")
            conversation['content'] = '\n'.join(parts)
            new_conversations.append(conversation)
        conversations = new_conversations
        
        input_dict = conversation_to_ids(conversations, tokenizer, llm_type, new_schema, max_length)
    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            print('images:',len(images),' image:',image.shape,'-->reshape_image:',reshape_image.shape)
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)

        input_dict["pixel_values"] = reshape_images
        input_dict["tgt_sizes"] = tgt_sizes

    else:
        input_dict["pixel_values"] = images
        input_dict["tgt_sizes"] = []

    return input_dict

def conversation_to_ids(conversation, tokenizer, llm_type=None, new_schema=False, max_length=2048):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    if llm_type == "llama3":
        input_ids, context, raw_msg = conversation_to_ids_llama3(
            conversation, tokenizer
        )
    elif llm_type == "qwen2":
        input_ids, context, raw_msg = conversation_to_ids_qwen2(
            conversation, tokenizer
        )
    else:
        input_ids, context, raw_msg = conversation_to_ids_minicpm(
            conversation, tokenizer
        )

    ids = torch.from_numpy(input_ids)
    context = torch.from_numpy(np.hstack(context))#, dtype=np.int8))

    if input_ids.shape[-1] > max_length:
        ids =ids[:max_length]
        context = context[:max_length]
        # logger.warning(f"The input length ({input_ids.shape[-1]}) exceeds the model's maximum length ({max_length}), so it has been truncated")
    
    if torch.all(context):
        # logger.error("No tokens available to compute loss.")
        raise Exception("No tokens available to compute loss.")

    # build target
    target = torch.full_like(ids, -100, dtype=torch.int32)
    
    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            if hasattr(tokenizer, "eot_id"):
                target[i - 1] = tokenizer.eot_id
            else:
                target[i - 1] = tokenizer.eos_id
    
    # build image bound
    if new_schema:
        start_cond = (ids == tokenizer.im_start_id) | (ids == tokenizer.slice_start_id)
        end_cond = (ids == tokenizer.im_end_id) | (ids == tokenizer.slice_end_id)
        image_start_tokens = torch.where(start_cond)[0]
        image_start_tokens += 1
        image_end_tokens = torch.where(end_cond)[0]
    else:
        image_start_tokens = torch.where(ids == tokenizer.im_start_id)[0]
        image_start_tokens += 1
        image_end_tokens = torch.where(ids == tokenizer.im_end_id)[0]
    if len(image_start_tokens) != len(image_end_tokens):
        logger.error("image start token != image end tokens")
        raise Exception("image start token != image end tokens")
    
    if len(image_start_tokens) > 0:
        image_bound = torch.hstack(
            [image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)]
        )
    else:
        image_bound = []

    position_ids = torch.arange(ids.size(0)).long()
    return {
        "input_ids": ids,
        "target": target,
        "image_bound": image_bound,
        "raw_msg": raw_msg,
        "position_ids": position_ids
    }

def conversation_to_ids_minicpm(conversation, tokenizer):
    # print('conversation_to_ids_minicpm-conversations:',conversation)
    raw_msg = ""
    input_ids = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "<用户>"
        else:
            prefix = "<AI>"
        # append eos
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix)[1:]  # remove bos
        message_ids = tokenizer.encode(message)[1:]

        input_ids.append(prefix_ids)
        input_ids.append(message_ids)

        context.append(np.ones((len(prefix_ids),), dtype=np.int8))
        if role == "assistant":
            context.append(np.zeros((len(message_ids),), dtype=np.int8))
        else:
            context.append(np.ones((len(message_ids),), dtype=np.int8))

        raw_msg += prefix + message

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)
    return input_ids, context, raw_msg

def conversation_to_ids_llama3(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    raw_msg = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False, chat_template=llama3_chat_template,
    )
    input_ids = np.array(input_ids)

    start_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    )[0]
    assistant_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[0]
    end_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    )[0]
    eot_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|eot_id|>"))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx in set((start_header_idxs + end_header_idxs) / 2):
            st = assistant_idx + 3  # assistant<|end_header_id|>\n\n
            for eot_idx in eot_idxs:
                if eot_idx > st:
                    context[st: eot_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)

    return input_ids, context, raw_msg

def conversation_to_ids_qwen2(conversation, tokenizer):
    raw_msg = ""
    chat = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "user"
        else:
            prefix = "assistant"
        chat.append({"role":prefix, "content":message})
        raw_msg += prefix + message
    assert set([i['role'] for i in chat]) & set(['assistant'])

    ret = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    input_ids = np.array(input_ids)

    start_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_start|>'))[0]
    assistant_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('assistant'))[0]
    end_idxs = np.where(input_ids == tokenizer.convert_tokens_to_ids('<|im_end|>'))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx-1 in set(start_idxs):
            st = assistant_idx + 1
            for end_idx in end_idxs:
                if end_idx > st:
                    context[st: end_idx + 1] = 0
                    break
                    
    input_ids = np.hstack(input_ids)
    context = np.hstack(context)
    return input_ids, context, raw_msg


class CCSBUAlignDataset(Dataset): # CaptionDataset
    def __init__(self, args, images ,labels, vis_processor, text_processor, split_name='test'):
        '''
        texts: np.array
        '''
        self.args = args
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.images = []
        self.labels = []
        # self.instruction= '[INST] <Img><ImageHere></Img> Describe this image in detail. [/INST]'
        
        self.instruction = '[INST] <Img><ImageHere></Img> Please provide a detailed description of the picture. [/INST] '

        # print(f'===== Dataset {split_name} =====')
        # print('self.vis_processor:',self.vis_processor)
        # print('self.text_processor:',self.text_processor)

        # print('self.instruction:',self.instruction)

        for i in range(len(images)):
            self.images.append( images[i] )
            self.labels.append( labels[i] )


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index] #torch.tensor(self.labels[index]).squeeze().to(self.args.device)
        image = self.images[index]
        text_input = self.instruction
        input_dict = {
            # 'prompt':text_input,
            'question':text_input,
            'image':image,#.to(self.args.device),
        }

        ### prepare target tokens
        self.args.tokenizer.padding_side = "right"
        label = label+'\n' 
        
        if self.args.model_type == 'T5':
            if self.args.tokenizer.bos_token_id == None:
                self.args.tokenizer.bos_token_id = 101 # set to 101 defaultly
            
            if self.split_name == 'train':
                input_dict['decoder_input_ids'] = input_dict['input_ids']
            else:
                bos_tensor = torch.tensor( [self.args.tokenizer.bos_token_id] ).to(input_dict['input_ids'].device)
                # input_dict['decoder_input_ids'] = torch.cat((bos_tensor, input_dict['input_ids'][1:]), dim=0)
                input_dict['decoder_input_ids'] = bos_tensor
        

        return input_dict, label


class LambadaDataset_LLM(Dataset):
    def __init__(self, args, texts ,labels, split_name='test'):
        '''
        texts: np.array
        '''
        self.args = args
        self.labels = []
        self.features = []
        self.input_dicts = []
        self.split_name = split_name

        if split_name == 'test':
            for i in range(len(texts)):
                ids = args.tokenizer(texts[i],return_tensors='pt')

                self.labels.append( args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                self.input_dicts.append(ids)

                # if i == 0:
                #     print('TEST TEXT:',texts[i])
                #     print('text_id:',ids['input_ids'].shape, ids['input_ids'])
                #     print('label:',self.labels[i] )
                #     print('-'*25)
        else:
            for i in range(len(texts)):
                ids = args.tokenizer(texts[i], \
                padding=args.padding,truncation=args.truncation ,\
                max_length=args.max_length,return_tensors='pt')

                self.labels.append(ids['input_ids'])#args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                self.input_dicts.append(ids)

                # if i == 0:
                #     print('TRAIN TEXT:',texts[i])
                #     print('text_id:',ids['input_ids'].shape, ids['input_ids'])
                #     print('label:',self.labels[i], args.tokenizer.convert_tokens_to_ids( labels[i] ) )
                #     print('-'*25)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance(input_dict[key_name], dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)
        
        if self.args.model_type == 'T5':
            if self.args.tokenizer.bos_token_id == None:
                self.args.tokenizer.bos_token_id = 101 # set to 101 defaultly
            
            if self.split_name == 'train':
                input_dict['decoder_input_ids'] = input_dict['input_ids']
            else:
                bos_tensor = torch.tensor( [self.args.tokenizer.bos_token_id] ).to(input_dict['input_ids'].device)
                # input_dict['decoder_input_ids'] = torch.cat((bos_tensor, input_dict['input_ids'][1:]), dim=0)
                input_dict['decoder_input_ids'] = bos_tensor


        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        # print('get item:',input_dict['decoder_input_ids'].shape,label.shape)
        return input_dict, label


class MMLUDataset_LLM(Dataset):
    def __init__(self, args, texts, labels, split_name='test'):
        '''
        texts: np.array
        '''
        self.args = args
        self.labels = []
        self.features = []
        self.input_dicts = []

        # if args.task_type == 'CausalLM':
        for i in range(len(texts)):
            ids = args.tokenizer(texts[i], \
                                 padding=args.padding, truncation=args.truncation, \
                                 max_length=args.max_length, return_tensors='pt')

            # if i == 0:
            #     print('TEXT:', texts[i])
            #     print('text_id:', ids['input_ids'].shape, ids['input_ids'])
            #     print('label:', labels[i], args.tokenizer.convert_tokens_to_ids(labels[i]))
            #     print('-' * 25)

            self.labels.append(args.tokenizer.convert_tokens_to_ids(labels[i]))
            self.input_dicts.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance(input_dict[key_name], dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label

class AlpacaDataset_LLM(Dataset):
    def __init__(self, args, sources, targets, split_name='train'):
        '''
        texts: np.array
        '''
        self.args = args
        self.split_name = split_name

        IGNORE_INDEX = args.tokenizer.pad_token_id  # -100

        def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
            """Tokenize a list of strings."""
                
            if split_name == 'train':
                tokenized_list = [
                    tokenizer(text, return_tensors="pt",
                        padding=args.padding, max_length=args.max_length,  truncation=args.truncation)
                    for text in strings]
            else:
                tokenized_list = [tokenizer(text,return_tensors="pt") for text in strings]


            input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
            attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]

            input_ids_lens = labels_lens = [
                tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
            ]

            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                input_ids_lens=input_ids_lens,
                labels_lens=labels_lens,
            )

        def preprocess(
                sources: Sequence[str],
                targets: Sequence[str],
                tokenizer,
        ) -> Dict:
            """Preprocess the data by tokenizing."""
            if split_name == 'train':
                examples = [s + t for s, t in zip(sources, targets)]
                examples_tokenized = _tokenize_fn(examples, tokenizer) # prompt+ans
                prompt_tokenized = _tokenize_fn(sources, tokenizer) # ans
                sources_tokenized = _tokenize_fn(targets, tokenizer) # ans
                
                input_ids = examples_tokenized["input_ids"]
                attention_mask = examples_tokenized["attention_mask"]
                
                labels = copy.deepcopy(input_ids)
                # for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]): # source_len:label length
                #     label[:source_len] = IGNORE_INDEX
                for label, source_len in zip(labels, prompt_tokenized["input_ids_lens"]): # source_len:label length
                    label[:source_len] = IGNORE_INDEX
                
            else:
                inputs_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in
                                                       (sources, targets)]
                input_ids = inputs_tokenized["input_ids"]
                attention_mask = inputs_tokenized["attention_mask"]
                labels = targets_tokenized["input_ids"]
            return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        data_dict = preprocess(sources, targets, args.tokenizer)

        self.input_dicts = [
            {'input_ids': data_dict['input_ids'][i], 'attention_mask': data_dict['attention_mask'][i]}
            for i in range(len(data_dict['input_ids'])) ] 
        self.labels = data_dict["labels"]  

 
        # print(f'=== Dataset Split = {split_name} ===')
        # for i in [0]:
        #     print('text:',self.input_dicts[i]['input_ids'].shape)
        #     print(self.args.tokenizer.decode(self.input_dicts[i]['input_ids'], skip_special_tokens=True))
        #     print('-'*25)
        #     print('label:')
        #     print(self.labels[i].shape, self.args.tokenizer.decode(self.labels[i],skip_special_tokens=True))
        #     print('='*50)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):
        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance( input_dict[key_name] , dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        if self.args.model_type == 'T5':
            if self.args.tokenizer.bos_token_id == None:
                self.args.tokenizer.bos_token_id = 101 # set to 101 defaultly
            
            if self.split_name == 'train':
                input_dict['decoder_input_ids'] = input_dict['input_ids']
            else:
                bos_tensor = torch.tensor( [self.args.tokenizer.bos_token_id] ).to(input_dict['input_ids'].device)
                # input_dict['decoder_input_ids'] = torch.cat((bos_tensor, input_dict['input_ids'][1:]), dim=0)
                input_dict['decoder_input_ids'] = bos_tensor
        
        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        
        return input_dict, label

class CNNDailyMailDataset(Dataset):
    def __init__(self, args, articles, highlights, split_name='train'):
        self.args = args
        self.split_name = split_name
        
        prompt_template = """Generate a summarization of the provided ducument.\n## Document: {article}\n## Summarization: """
        # prompt_template = """Generate a summarization of the provided ducument.\n## Document: {article}\n## Summarization: {highlight}"""
        
        if split_name == 'train':
            self.input_dicts = []
            self.labels = []
            for a,h in zip(articles,highlights):
                article_input_text = prompt_template.format(article=a)
                article_input_ids = self.args.tokenizer(article_input_text, return_tensors="pt").input_ids
                article_len = article_input_ids.shape[-1]
                print('article_input_ids:',article_input_ids.shape)
                
                highlight_input_ids = self.args.tokenizer(h, return_tensors="pt").input_ids
                print('highlight_input_ids:',highlight_input_ids.shape)
                
                combined_input_ids = torch.cat((article_input_ids, highlight_input_ids), dim=1)
                print('combined_input_ids:',combined_input_ids.shape)
                
                pad_length = max(0,args.max_length - combined_input_ids.shape[-1])
                combined_input_ids = torch.nn.functional.pad(combined_input_ids, (0, pad_length), value=self.args.tokenizer.pad_token_id)
                # combined_input_ids = torch.nn.utils.rnn.pad_sequence(combined_input_ids, batch_first=True, padding_value=self.args.tokenizer.pad_token_id)
                attention_mask=combined_input_ids.ne(self.args.tokenizer.pad_token_id)
                self.input_dicts.append(
                    dict(input_ids = combined_input_ids, attention_mask = attention_mask)
                )
                
                label_ids = copy.deepcopy(combined_input_ids)
                label_ids[0,:article_len]=self.args.tokenizer.pad_token_id
                pad_length = max(0,args.max_length - label_ids.shape[-1])
                label_ids = torch.nn.functional.pad(label_ids, (0, pad_length), value=self.args.tokenizer.pad_token_id)
                # label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=self.args.tokenizer.pad_token_id)
                self.labels.append(label_ids)
                
        else:
            self.input_dicts = []
            self.labels = []
            for a,h in zip(articles,highlights):
                input_text = prompt_template.format(article=a)
                self.input_dicts.append( self.args.tokenizer(input_text, return_tensors="pt") )
                self.labels.append( self.args.tokenizer(h, return_tensors="pt").input_ids )
        
        print(f'=== Dataset Split = {split_name} ===')
        print(len(self.input_dicts),len(self.labels))
        for i in range(2):
            print('text:',self.input_dicts[i]['input_ids'].shape)
            print(self.args.tokenizer.batch_decode(self.input_dicts[i]['input_ids']))
            print('-'*25)
            print('label:',self.labels[i].shape)
            print(self.args.tokenizer.batch_decode(self.labels[i]))
            print('='*50)
        assert 1>2
                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):
        input_dict = self.input_dicts[item_idx]
        for key_name in input_dict.keys():
            if not isinstance( input_dict[key_name] , dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)
        
        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label


class GSMDataset_LLM(Dataset):
    def __init__(self, args, qns, ans, split_name='train', loss_on_prefix=True):
        self.args=args
        self.qns = qns
        self.ans = ans
        self.loss_on_prefix = loss_on_prefix
        self.split_name = split_name
        if split_name == 'train':
            self.input_dicts = [ self.args.tokenizer(
                                    self.qns[i]+self.ans[i],return_tensors="pt",
                                    padding=args.padding,#"longest",
                                    max_length=args.max_length, #tokenizer.model_max_length,
                                    truncation=args.truncation, #True,
                                )
                for i in range(len(self.ans))
            ]
            self.labels = [ _input['input_ids'] for _input in self.input_dicts ]

        else:
            self.input_dicts = [ self.args.tokenizer(
                                    self.qns[i],return_tensors="pt",
                                    # padding=args.padding,#"longest",
                                    # max_length=args.max_length, #tokenizer.model_max_length,
                                    # truncation=args.truncation, #True,
                                )
                for i in range(len(self.ans))
            ]
            self.labels = [ self.args.tokenizer(
                                    self.ans[i],return_tensors="pt",
                                    # padding=args.padding,#"longest",
                                    # max_length=args.max_length, #tokenizer.model_max_length,
                                    # truncation=args.truncation, #True,
                                )['input_ids']
                for i in range(len(self.ans))
            ]
            

    def __len__(self):
        return len(self.ans)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance( input_dict[key_name] , dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        if self.args.model_type == 'T5':
            if self.args.tokenizer.bos_token_id == None:
                self.args.tokenizer.bos_token_id = 101 # set to 101 defaultly
            
            if self.split_name == 'train':
                input_dict['decoder_input_ids'] = input_dict['input_ids']
                # input_dict['label'] = self.labels[item_idx].to(self.args.device)
            else:
                bos_tensor = torch.tensor( [self.args.tokenizer.bos_token_id] ).to(input_dict['input_ids'].device)
                input_dict['decoder_input_ids'] = bos_tensor
        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        
        # print('--get item decoder_input_ids:',input_dict['decoder_input_ids'].shape,input_dict['decoder_input_ids'])
        # print('--get item label:',input_dict['label'].shape,input_dict['label'])
        return input_dict, label

   
class MATHDataset_LLM(Dataset):
    def __init__(self, args, sources ,targets, split_name='train'):
        '''
        texts: np.array
        '''
        self.args = args
        self.split_name = split_name
        
        IGNORE_INDEX = args.tokenizer.pad_token_id #-100
        def _tokenize_fn(strings: Sequence[str], tokenizer) -> Dict:
            """Tokenize a list of strings."""

            tokenized_list = [
                tokenizer(
                    text,
                    return_tensors="pt",
                    padding=args.padding,#"longest",
                    max_length=args.max_length, #tokenizer.model_max_length,
                    truncation=args.truncation, #True,
                )
                for text in strings
            ]

            input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
            attention_mask = [tokenized.attention_mask[0] for tokenized in tokenized_list]

            input_ids_lens = labels_lens = [
                tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
            ]

            return dict(
                input_ids=input_ids,
                attention_mask = attention_mask,
                labels=labels,
                input_ids_lens=input_ids_lens,
                labels_lens=labels_lens,
            )

        def preprocess(
            sources: Sequence[str],
            targets: Sequence[str],
            tokenizer,
        ) -> Dict:
            """Preprocess the data by tokenizing."""
            if split_name == 'train':
                examples = [s + t for s, t in zip(sources, targets)]
                examples_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, targets)]

                input_ids = examples_tokenized["input_ids"]
                attention_mask = examples_tokenized["attention_mask"]

                labels = copy.deepcopy(input_ids)
                for label, target_len in zip(labels, targets_tokenized["input_ids_lens"]):
                    label[:-target_len] = IGNORE_INDEX
            else:
                inputs_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (sources, targets)]

                input_ids = inputs_tokenized["input_ids"]
                attention_mask = inputs_tokenized["attention_mask"]
                labels = targets_tokenized["input_ids"]
            # input_ids: prompt +  target
            # label: masked_prompt + target
            return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        data_dict = preprocess(sources, targets, args.tokenizer)

        self.input_dicts = [
                {'input_ids':data_dict['input_ids'][i], 'attention_mask':data_dict['attention_mask'][i]}
                for i in range( len(data_dict['input_ids']) )
            ] # list of input_dicts
        self.labels = data_dict["labels"] # list of tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item_idx):

        input_dict = self.input_dicts[item_idx]

        for key_name in input_dict.keys():
            if not isinstance(input_dict[key_name], dict):
                input_dict[key_name] = torch.tensor(input_dict[key_name]).squeeze().to(self.args.device)

        if self.args.model_type == 'T5':
            if self.args.tokenizer.bos_token_id == None:
                self.args.tokenizer.bos_token_id = 101 # set to 101 defaultly
            
            if self.split_name == 'train':
                input_dict['decoder_input_ids'] = input_dict['input_ids']
            else:
                bos_tensor = torch.tensor( [self.args.tokenizer.bos_token_id] ).to(input_dict['input_ids'].device)
                # input_dict['decoder_input_ids'] = torch.cat((bos_tensor, input_dict['input_ids'][1:]), dim=0)
                input_dict['decoder_input_ids'] = bos_tensor
          
        label = torch.tensor(self.labels[item_idx]).squeeze().to(self.args.device)
        return input_dict, label


class PassiveDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i = self.data[item_idx]
        return torch.tensor(data_i, dtype=torch.float32), torch.tensor([] * data_i.size()[0])


class ActiveDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i, target_i = self.data[item_idx], self.labels[item_idx]
        return torch.tensor(data_i.clone().detach(), dtype=torch.float32), torch.tensor(target_i.clone().detach(),
                                                                                        dtype=torch.long)


class SimpleTwoPartyDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, data_a, data_b, labels):
        self.data_a = data_a
        self.data_b = data_b
        self.labels = labels

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, item_idx):
        data_a_i, data_b_i, target_i = self.data_a[item_idx], self.data_b[item_idx], self.labels[item_idx]
        return (torch.tensor(data_a_i).float(), torch.tensor(data_b_i).float()), \
            torch.tensor(target_i.numpy(), dtype=torch.long)


def get_dataloaders(train_dataset: SimpleTwoPartyDataset, valid_dataset: SimpleTwoPartyDataset, batch_size=32,
                    num_workers=1):
    mnist_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mnist_valid_loader = None
    if valid_dataset is not None:
        mnist_valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=True, num_workers=num_workers)
    return mnist_train_loader, mnist_valid_loader
