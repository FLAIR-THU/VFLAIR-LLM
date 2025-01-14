from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
from collections import Counter,defaultdict
import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_customized_mapping(train_data, eps, top_k):
    tokenized_sentences = [simple_preprocess(sentence) for sentence in train_data]
    corpus = " ".join(train_data)
    word_freq = [x[0] for x in Counter(corpus.split()).most_common()]

    vector_size = 300
    model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size, window=5, min_count=1, workers=4)
    model.train(train_data, total_examples=1, epochs=10)


    # Get embeddings for each word
    embeddings = []
    idx2word = []
    word2idx = {}
    for i, word in enumerate(model.wv.index_to_key):
        embedding = model.wv[word]
        embeddings.append(embedding)
        idx2word.append(word)
        word2idx[word] = i

    embeddings = np.array(embeddings)
    idx2word = np.asarray(idx2word)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = np.asarray(embeddings / norm, "float64")
    
    
    word_hash = defaultdict(str)
    sim_word_dict = defaultdict(list)
    p_dict = defaultdict(list)
    for i in range(len(word_freq)):
        word = word_freq[i]
        if word in word2idx:
            if word not in word_hash:
                index_list = np.dot(embeddings[word2idx[word]], embeddings.T).argsort()[::-1][:top_k]
                word_list = [idx2word[x] for x in index_list]
                embedding_list = np.array([embeddings[x] for x in index_list])
                
                for x in word_list:
                    if x not in word_hash:
                        word_hash[x] = word
                        sim_dist_list = np.dot(embeddings[word2idx[x]], embedding_list.T)
                        min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                        min_dist = min(sim_dist_list)
                        new_sim_dist_list = [(x-min_dist)/min_max_dist for x in sim_dist_list]
                        tmp = [np.exp(eps*x/2) for x in new_sim_dist_list]
                        norm = sum(tmp)
                        p = [x/norm for x in tmp]
                        p_dict[x] = p
                        sim_word_dict[x] =  word_list
                inf_embedding = [0] * vector_size
                for i in index_list:
                    embeddings[i,:] = inf_embedding
    
    return p_dict, sim_word_dict


def generate_new_sents_s1(dataset,sim_word_dict,p_dict,save_stop_words,args):
    # print('origin dataset')
    # print(type(dataset),len(dataset))
    # print(dataset[:3])
    punct = list(string.punctuation)

    
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        nltk.download('punkt')
        stop_words = set(stopwords.words('english'))
    
    cnt = 0 
    raw_cnt = 0 
    stop_cnt = 0 
    new_dataset = []

    for i in range(len(dataset)):
        record = dataset[i].split()
        new_record = []
        for word in record:
            if args.dataset in ['GMS8K','GMS8K-test','MATH']:
                if word in ['one','two','three','four','five','seven','eight','nine','ten']:
                    new_record.append(word)
                    continue
                
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict):
                if word in stop_words:
                    stop_cnt += 1  
                    raw_cnt += 1   
                
                if args.dataset not in ['GMS8K','GMS8K-test','MATH']:
                    if is_number(word):
                        try:
                            word = str(round(float(word))+np.random.randint(1000))
                        except:
                            pass                   
                new_record.append(word)
            else:
                p = p_dict[word]
                new_word = np.random.choice(sim_word_dict[word],1,p=p)[0]
                new_record.append(new_word)
                if new_word == word:
                    raw_cnt += 1 

            cnt += 1 
        new_dataset.append(" ".join(new_record))
    new_dataset = np.array(new_dataset)
    
    # print('new dataset')
    # print(type(new_dataset),len(new_dataset))
    # print(new_dataset[:3])
    # df.sentence = new_dataset

    # if not os.path.exists(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}"):
    #     os.mkdir(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}")
    # if type == "train":
    #     df.to_csv(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/train.tsv","\t",index=0)
    # else:
    #     df.to_csv(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/test.tsv","\t",index=0)

    return new_dataset