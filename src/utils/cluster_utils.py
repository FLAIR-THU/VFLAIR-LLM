from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
import torch
import json
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, default_data_collator, DataCollatorWithPadding, DataCollatorForTokenClassification
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import os

def generate_token_embeddings(model, dataloader, use_subword=True):
    total_tokens = {}
    token_cnt = {}
    model.eval()
    pro_bar = tqdm(range(len(dataloader)))
    step=0
    
    hidden_size = model.config.hidden_size
    device = model.device
    pro_bar.set_description('generating contextual embedding')
    candicate_embedding = torch.zeros(model.config.vocab_size, hidden_size, device=device)
    token_cnt = torch.zeros(model.config.vocab_size, device=device)
    for batch_data in dataloader:
        step+=1
        for batch in batch_data:
            batch = batch[0]
            for key in batch.keys():
                if batch[key] != None:
                    batch[key] = batch[key].unsqueeze(0)
            
            with torch.no_grad():
                batch = {key:value.to(device) for key,value in batch.items()}
                batch['output_hidden_states'] = True
                if 'special_tokens_mask' in batch:
                    special_tokens_mask = batch.pop('special_tokens_mask')
                if 'labels' in batch:
                    batch.pop('labels')
                outputs = model(**batch)
                input_ids = batch['input_ids']
                # target_hidden_states = outputs.get('hidden_states')[2]
                # print('target_hidden_states:',target_hidden_states.shape)
                target_hidden_states = outputs.get('inputs_embeds')
                
                attention_mask = batch['attention_mask'] 
                if use_subword:
                    valid_ids = (attention_mask!=0) 
                else:
                    valid_ids = (attention_mask!=0) & (special_tokens_mask!=0)
                
                input_ids = input_ids[valid_ids]
                target_hidden_states = target_hidden_states[valid_ids]
                
                candicate_embedding.scatter_add_(0, input_ids.unsqueeze(1).repeat(1, 768), target_hidden_states)
                token_cnt.scatter_add_(0, input_ids, torch.ones(input_ids.shape).type_as(token_cnt))
            
            pro_bar.update(1)
    # print('token_cnt:',token_cnt.shape) # 30522
    valid_token_embeddings_ids = token_cnt.nonzero().squeeze() #598
    token_embeddings = candicate_embedding[valid_token_embeddings_ids]
    token_embeddings = (token_embeddings/token_cnt[valid_token_embeddings_ids].unsqueeze(1)).cpu()
    sample2vocab = {sample_ids:vocab_ids.item() for sample_ids, vocab_ids in enumerate(valid_token_embeddings_ids)}
    # print('valid_token_embeddings_ids:',valid_token_embeddings_ids.shape,valid_token_embeddings_ids)
    
    return token_embeddings, sample2vocab

def generate_hierachy_center(cluster_results, contextual_embedding, cluster_num):
    cluster_center = torch.zeros((cluster_num, contextual_embedding.shape[1]))
    token_cnt = torch.zeros(cluster_num)
    
    for sample_ids, cluster_ids in enumerate(cluster_results):
        cluster_center[cluster_ids] += contextual_embedding[sample_ids]
        token_cnt[cluster_ids] += 1
    
    for i in range(cluster_num):
        cluster_center[i] /= token_cnt[i]
    
    return cluster_center
        

import time
def run_cluster(model, dataloader, LR_word=[],  cluster_num=100, cluster_method='kmeans', use_subword=True):
    token_embeddings, sample2vocab = generate_token_embeddings(model, dataloader, use_subword=use_subword)
    
    # token_embeddings: 598, embeddim768
    # sample2vocab a lis of 598 dicts sample_id:vocab_id
    print(f'run {cluster_method} clustering...')
    start_time = time.time()
    if cluster_method == 'kmeans':
        clusters = KMeans(n_clusters=cluster_num, random_state=0).fit(token_embeddings)
        cluster_results = clusters.predict(token_embeddings)
        cluster_center = torch.tensor(clusters.cluster_centers_).to(model.device)
    elif cluster_method == 'hierarchy':
        clusters = AgglomerativeClustering(n_clusters=cluster_num).fit(token_embeddings)
        cluster_results = clusters.labels_
        cluster_center = generate_hierachy_center(cluster_results, token_embeddings, cluster_num)
    elif cluster_method == 'gmm':
        from sklearn.mixture import GaussianMixture
        clusters = GaussianMixture(n_components=cluster_num, random_state=0).fit(token_embeddings)
        cluster_results = clusters.predict(token_embeddings)
        cluster_center = torch.tensor(clusters.means_).to(model.device)
    print(f'run {cluster_method} clustering...done! cost {time.time()-start_time}')
    token2cluster = {}
    for sample_ids, cluster_ids in enumerate(cluster_results):
        token2cluster[sample2vocab[sample_ids]] = int(cluster_ids.item())
    del(token_embeddings)
    del(sample2vocab)
    return token2cluster, cluster_center#, token_embeddings, sample2vocab

def redivide_cluster(label_related_words, cluster_center, tokenizer, token2cluster, token_embeddings, embeddings_to_vocabulary, cluster_num):
    label_related_words_ids = [tokenizer.convert_tokens_to_ids(item) for item in label_related_words]
    
    valid_token_embeddings_ids = []
    for _key in token2cluster.keys():
        valid_token_embeddings_ids.append(_key)
    valid_token = tokenizer.convert_ids_to_tokens(valid_token_embeddings_ids)
    
    LRWords_cluster = [[token2cluster[token_id] for token_id in label_i] for label_i in label_related_words_ids]
    LRWords_cluster = torch.tensor(LRWords_cluster)
    
    label_num, topk = LRWords_cluster.shape
    vocab_to_embeddings = {int(value):int(key) for key,value in embeddings_to_vocabulary.items()}
    
    for i in range(label_num):
        label_i_topk = LRWords_cluster[i]
        label_i_clusters = set(label_i_topk.tolist())
        for j in range(i+1, label_num):
            label_j_topk = LRWords_cluster[j]
            conflit_mask = (label_i_topk.unsqueeze(1) == label_j_topk.unsqueeze(0))
            conflit_pairs = conflit_mask.nonzero()

            for conflit_item in conflit_pairs:
                token_ids = label_related_words_ids[j][conflit_item[1]]
                token_embedding = token_embeddings[vocab_to_embeddings[token_ids]]
                candicate_cluster = torch.topk(torch.cdist(token_embedding.unsqueeze(0).type_as(cluster_center), cluster_center), k=min(30, cluster_num))[1].squeeze()
                for candicate in candicate_cluster:
                    candicate = candicate.item()
                    if candicate not in label_i_clusters:
                        token2cluster[token_ids] = candicate
                        break
    ori_cluster_center = cluster_center.cpu().detach().clone()
    cluster_center =  [[] for i in range(cluster_num)]
    for token_id, cluster_id in token2cluster.items():
        if token_id == 1:
            continue
        cluster_center[cluster_id].append(token_embeddings[vocab_to_embeddings[token_id]])
    for cluster_id in range(cluster_num):
        cluster_center[cluster_id] = torch.stack(cluster_center[cluster_id]).mean(dim=0)
    cluster_center = torch.stack(cluster_center)
    print(torch.nn.functional.cosine_similarity(ori_cluster_center, cluster_center))
    return token2cluster, cluster_center

def load_cluster_results(data_dir):
    with open(f'{data_dir}/token2cluster.json', "r") as f:
        token2cluster = json.load(f)

    cluster_center = None
    cluster_center_file = f'{data_dir}/cluster_center.pt'
    if os.path.exists(cluster_center_file):
        cluster_center = torch.load(cluster_center_file)

    return token2cluster, cluster_center

def save_cluster_results(token2cluster, cluster_center, data_dir, prefix=''):
    unique, counts = np.unique(list(token2cluster.values()), return_counts=True)
    plt.figure(figsize=(25, 10))
    plt.bar(unique, counts)
    plt.title(f'result statistics')
    plt.xlabel('cluster center index')
    plt.ylabel('num of tokens')
    plt.savefig(f'{data_dir}/vis.png')
    with open(f'{data_dir}/token2cluster.json',"w") as f:
        f.write(json.dumps(token2cluster))
    if cluster_center != None:
        torch.save(cluster_center.squeeze(), f'{data_dir}/cluster_center.pt')

class CenterLoss(nn.Module):
    def __init__(self, cluster_num,hidden_size, w_cluster_close, w_cluster_away):
        super().__init__()
        self.w_cluster_close = w_cluster_close
        self.w_cluster_away = w_cluster_away
        self.cluster_embedding = nn.Embedding(cluster_num, hidden_size)
        self.loss_fct = torch.nn.MSELoss()
    
    def forward(self, clean_hidden_states, cluster_ids):
        """
        input 
            clean_hidden_states (sample_num, hidden_size)
            cluster_ids (sample_num)
        
        output:
            center_loss_dict
        """
        
        target_cluster = self.cluster_embedding(cluster_ids)

        center_loss = self.loss_fct(clean_hidden_states, target_cluster)                  
        
        cluster_center = self.cluster_embedding.weight
        cluster_num = cluster_center.shape[0]
        
        cluster_away_loss =  - torch.triu(torch.cdist(cluster_center, cluster_center), diagonal=1).sum()/((1+cluster_num)*cluster_num/2)
            
        center_loss_dict = {'center_loss': self.w_cluster_close * center_loss, 
                            'cluster_away_loss': self.w_cluster_away * cluster_away_loss}
        return center_loss_dict

  
if __name__ == '__main__':
    task_name = 'sst2'
    model_name='roberta-base'
    target_layer = 3
    # kmeans gmm hierarchy
    cluster_method = 'kmeans'
    cluster_num = 100
    use_subword = True
    
    cluster_pipeline(task_name=task_name, model_name=model_name, target_layer=target_layer, cluster_method=cluster_method, cluster_num=cluster_num, use_subword=use_subword)
