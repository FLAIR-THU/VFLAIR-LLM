import json
import string
# import tiktoken
import random
from sklearn.metrics.pairwise import euclidean_distances
import tqdm
from decimal import getcontext
import numpy as np
import json
from transformers import GPT2Tokenizer
import os 
import tqdm
import re
getcontext().prec = 100
#os.environ["http_proxy"] = "http://127.0.0.1:10809"
#os.environ["https_proxy"] = "http://127.0.0.1:10809"


# def get_first_50_tokens(text):
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     tokens = tokenizer.tokenize(text)
#     first_50_tokens = tokens[:50]
#     tokenized_string = tokenizer.convert_tokens_to_string(first_50_tokens)
#     return tokenized_string

# def get_first_100_tokens(text):
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     tokens = tokenizer.tokenize(text)
#     first_100_tokens = tokens[:100]
#     tokenized_string = tokenizer.convert_tokens_to_string(first_100_tokens)
#     return tokenized_string


## new
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def add_laplace_noise_to_vector(vector, epsilon, delta_f_new):
    vector = np.asarray(vector, dtype=np.longdouble)
    if epsilon == 0:
        beta_values = delta_f_new * 0
    else:
        beta_values = delta_f_new / (0.5 * epsilon)
    noise = np.random.laplace(loc=0, scale=beta_values, size=len(beta_values))
    noisy_vector = vector + noise

    return noisy_vector




def contains_english_chars(string):
    pattern = r'[a-zA-Z]'
    match = re.search(pattern, string)
    return bool(match)

def contains_non_english_chars(string):
    pattern = r'[^a-zA-Z]'
    match = re.search(pattern, string)
    return bool(match)


def filter_tokens(token2index):
    filtered_index2token = {}
    for key, idx in tqdm.tqdm(token2index.items()):
       
        if key.startswith('<'):
            continue
        # if not key.startswith('▁'):
        #     continue
        val_ = key.replace("▁", "")
        # if val_ == val_.upper():
        #     continue
        if contains_non_english_chars(val_):
            continue
        if 3 < len(val_) < 16 and contains_english_chars(val_):
            filtered_index2token[idx] = key

    return filtered_index2token


def create_sensitivity_of_embeddings(all_embedding_matrix):
    n_dimensions = all_embedding_matrix.shape[1]
    delta_f_new = np.zeros(n_dimensions)
    for dim in tqdm.trange(n_dimensions):
        dim_data = all_embedding_matrix[:, dim]
        sorted_dim_data = np.sort(dim_data)
        differences = sorted_dim_data[-1] - sorted_dim_data[0]
        delta_f_new[dim] = differences
    return delta_f_new

def create_sorted_embedding_matrix(token_list, similarity_matrix):
    token_2_sorted_distances = dict()
    token_array = np.array(token_list)
    for idx, token in tqdm.tqdm(enumerate(token_list)):
        similarity_array = similarity_matrix[idx]
        sorted_indices = np.argsort(similarity_array)[::-1]
        token_2_sorted_distances[token] = [token_array[sorted_indices].tolist(), similarity_array[sorted_indices].tolist()]
    return token_2_sorted_distances

def cosine_similarity_vectors(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def generate_inferdpt_kit(embedding_matrix,token_list):
    def cosine_simi(embedding_matrix1, embedding_matrix2):
        dot_product = np.dot(embedding_matrix1, embedding_matrix2.T)
        norm_matrix1 = np.linalg.norm(embedding_matrix1, axis=1)
        norm_matrix2 = np.linalg.norm(embedding_matrix2, axis=1)
        similarity = dot_product / (np.outer(norm_matrix1, norm_matrix2))
        return similarity
    assert len(embedding_matrix) == len(token_list)
    similarity_matrix = cosine_simi(embedding_matrix, embedding_matrix)
    
    sorted_similarities = create_sorted_embedding_matrix(token_list, similarity_matrix) # dict 20348
    # print('sorted_similarities:',type(sorted_similarities),len(sorted_similarities))
    
    delta_f = create_sensitivity_of_embeddings(embedding_matrix) # (768,)
    # print('delta_f:',type(delta_f),delta_f.shape) 
    
    token_to_vector_dict = {}
    for token, embedding in zip(token_list, embedding_matrix):
        token_to_vector_dict[token] = embedding
    token_to_vector_dict = token_to_vector_dict
    print('token_to_vector_dict:',len(token_to_vector_dict))
    
    return token_to_vector_dict, sorted_similarities, delta_f
