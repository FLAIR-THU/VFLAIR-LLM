import os
import re
import torch

class MATHEval:
    def __init__(self,args):
        self.args = args
        
    def evaluate(self, predict_word_list, target_word_list):
        def wash(token_id_list, washed_ids):
            washed_token_id_list = []
            for token_ids in token_id_list:
                token_ids = list(token_ids)
                for washed_id in washed_ids:
                    while washed_id in token_ids:
                        token_ids.remove(washed_id)
                
                washed_token_id_list.append(torch.tensor(token_ids) )
            return washed_token_id_list

        washed_ids = [self.args.tokenizer.pad_token_id, self.args.tokenizer.eos_token_id, self.args.tokenizer.bos_token_id]
        predict_word_list = wash(predict_word_list,washed_ids )

        predict_word_list = [
            self.args.tokenizer.decode(_ids)
            for _ids in list(predict_word_list)]

        target_word_list = [
            self.args.tokenizer.decode(_ids)
            for _ids in list(target_word_list)]
                        
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
                        
        def process_results(completion, answer): # doc
            split_ans = completion.split('The answer is: ')
            if len(split_ans) > 1:
                ans = split_ans[-1]
                extract_ans_temp = ans.split('.\n')[0]
                extract_ans_temp = extract_ans_temp.strip()
                if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
                    extract_ans = extract_ans_temp[0:-1]
                else:
                    extract_ans = extract_ans_temp
                extract_ans = extract_ans.strip()
                
                if is_equiv(extract_ans, answer):
                    return True
                else:
                    return False
            else:
                return False

        results = []
        for i in range(len(target_word_list)):
            res = process_results(predict_word_list[i],target_word_list[i])
            results.append(res)
        acc = sum(results) / len(results)
        return acc
                    
