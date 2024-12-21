import itertools
import json
import os
import re
from collections import namedtuple

import torch
from tqdm import tqdm
from fraction import Fraction
class GMS8KEval:
    def __init__(self,args):
        self.args = args
        
    
    def evaluate(self, predict_word_list, target_word_list):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                pass
            # try:
            #     import unicodedata
            #     unicodedata.numeric(s)
            #     return True
            # except (TypeError, ValueError):
            #     pass
            return False

        def extract_answer_number(completion):
            text = completion.split('The answer is: ')
            if len(text) > 1:
                extract_ans = text[-1].strip()
                match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
                if match:
                    if '/' in match.group():
                        denominator = match.group().split('/')[1]
                        numerator = match.group().split('/')[0]
                        if is_number(denominator) == True and is_number(numerator) == True:
                            if denominator == '0':
                                return round(float(numerator.replace(',', '')))
                            else:
                                frac = Fraction(match.group().replace(',', ''))
                                num_numerator = frac.numerator
                                num_denominator = frac.denominator
                                return round(float(num_numerator / num_denominator))
                        else:
                            return None
                    else:
                        if float(match.group().replace(',', '')) == float('inf'):
                            return None
                        return round(float(match.group().replace(',', '')))
                else:
                    return None
            else:
                return None
        
        def wash(token_id_list, washed_ids):
            washed_token_id_list = []
            for token_ids in token_id_list:
                token_ids = token_ids.tolist()
                if not isinstance(token_ids, list):
                    token_ids = [token_ids]
                for _id in washed_ids:
                    while _id in token_ids:
                        token_ids.remove(_id)
                washed_token_id_list.append(torch.tensor(token_ids) )
            return washed_token_id_list

        def is_equiv(str1, str2, verbose=False):
            if str1 is None and str2 is None:
                print("WARNING: Both None")
                return True
            if str1 is None or str2 is None:
                return False

            try:
                ss1 = strip_string(str1)
                ss2 = strip_string(str2)
                #pdb.set_trace()
                if verbose:
                    print(ss1, ss2)
                return ss1 == ss2
            except Exception:
                return str1 == str2
       
        washed_ids = [self.args.tokenizer.pad_token_id, self.args.tokenizer.eos_token_id, self.args.tokenizer.bos_token_id]
        predict_word_list = wash(predict_word_list,washed_ids )
        # target_word_list = wash(target_word_list,washed_ids )

        predict_word_list = [
            self.args.tokenizer.decode(_ids)
            for _ids in list(predict_word_list)]

        target_word_list = [
            self.args.tokenizer.decode(_ids)
            for _ids in list(target_word_list)]
        
        results = []
        for i in range(len(target_word_list)):
            pred_ans = str(extract_answer_number(predict_word_list[i]))
            
            
            res = is_equiv(pred_ans,target_word_list[i])
            # print('-'*100)
            # print('PRED:',predict_word_list[i])
            # print('Extract PRED:',type(pred_ans), pred_ans)
            # print('GOLD:',type(target_word_list[i]),target_word_list[i])
            # print('SCORE:',res)
            
            results.append(res)
        acc = sum(results) / len(results)
        return acc
