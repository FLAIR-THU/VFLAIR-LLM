from collections import Counter
import numpy as np
import random

def get_vocab(dst, tokenizer):
    '''
    dst:( X, y ) 
    X: [sample1, sample2...]
    '''
    vocab=Counter()
    docs =  [_d[0] for _d in dst]
    
    for text in docs:
        text = tokenizer.decode(text['input_ids'],skip_special_tokens=True)
        # print('text:',text)
        tokenized_text = tokenizer.tokenize(text)
        for token in tokenized_text:
            vocab[token]+=1
    return vocab

def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
    from scipy.special import softmax
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix

def SanText_plus(doc, word2id, sword2id, all_words, prob_matrix, p):
    
    id2sword = {v: k for k, v in sword2id.items()}
    new_doc = []
    for word in doc:
        if word in word2id:
            # In-vocab
            if word in sword2id:
                #Sensitive Words
                index = word2id[word]
                sampling_prob = prob_matrix[index]
                sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                new_doc.append(id2sword[sampling_index[0]])
            else:
                #Non-sensitive words
                flip_p=random.random()
                if flip_p<=p:
                    #sample a word from Vs based on prob matrix
                    index = word2id[word]
                    sampling_prob = prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    new_doc.append(id2sword[sampling_index[0]])
                else:
                    #keep as the original
                    new_doc.append(word)
        else:
            #Out-of-Vocab words
            sampling_prob = 1 / len(all_words) * np.ones(len(all_words), )
            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
            new_doc.append(all_words[sampling_index[0]])

    new_doc = " ".join(new_doc)
    return new_doc