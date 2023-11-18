# %%
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
import time
import json
import re
# %%

json_file_path = 'maud_squad_train.json'

with open(json_file_path, 'r') as json_file:
    # Load and parse the JSON data
    json_data = json.load(json_file)
    
data = json_data['data'][0]
title = data['title']
paras = data['paragraphs'][0]
qas, context = paras['qas'], paras['context']

# %%
def linear_search_fuzzy(context, answer, big_stride = None, small_stride = None):
    if big_stride == None:
        big_stride = len(context)//15
    if small_stride == None:
        small_stride = len(answer)//5
    
    # Split context into overlapping sections of len(context)//10 and stride len(context)//15
    window_size = len(context)//10
    best_ratio = 0
    start_inds = list(range(0,len(context) - window_size + 1, big_stride))
    for i in start_inds[:-1]:
        sub_window = context[i:i+window_size]
        window_ratio = fuzz.partial_ratio(sub_window, answer)
        
        if window_ratio > best_ratio:
            best_ratio = window_ratio
            best_window = sub_window
            
    sub_window = context[start_inds[-1]:]
    window_ratio = fuzz.partial_ratio(sub_window, answer)
    if window_ratio > best_ratio:
        best_ratio = window_ratio
        best_window = sub_window
        
    la = len(answer)
    window_size = la
    best_ratio = 0
    frs = []

    for start in range(0,len(best_window)-window_size+1, small_stride):
        context_string = best_window[start - len(answer)//4 : start+window_size + len(answer)//4]
        window_ratio = fuzz.partial_ratio(context_string, answer)
        
        if window_ratio > best_ratio:
            best_ratio = window_ratio
            best_sub_window = context_string

    return best_sub_window


def binary_search_fuzzy(context, answer):
    start = 0
    end = len(context)
    
    thres = 90
    
    left_ratios = []
    right_ratios = []
    
    depth = 0
    
    while (end - start > 1.6*len(answer)) and (depth < 100):
        depth += 1
        mid = (start+end)//2
        left = context[start:mid]
        right = context[mid:end]
        
        left_ratio = fuzz.partial_ratio(left, answer)
        right_ratio = fuzz.partial_ratio(right, answer)
        
        left_ratios.append(left_ratio)
        right_ratios.append(right_ratio)
        
        # If we found the window
        if left_ratio - right_ratio > thres:
            return (left_ratios, right_ratios, start, left)
        
        elif right_ratio - left_ratio > thres:
            return (left_ratios, right_ratios, mid, right)
        
        # Left half vs Right half
        if left_ratio > right_ratio:
            end = mid + len(answer)//2 + 10
        else:
            start = mid - len(answer)//2 - 10
    return (left_ratios, right_ratios, start, context[start:end])


def cut_window(window, answer, answer_words):
    thres = 90
    answer_words = answer.split()
    for i in range(len(answer_words)):
        word = answer_words[i]
        ind = window.find(word)
        flag = False
        if ind != -1:
            start = ind
            for j in range(i):
                start -= len(answer_words[j]) + 1
            if fuzz.partial_ratio(answer, window[start:start + len(answer) + 10]) > thres:
                flag = True
                break
        if flag == True:
            break
    best_ratio = 0
    # answer_words = answer[::-1].split()
    # for i in range(len(answer_words)):
    #     word = answer_words[i]
    #     ind = window[::-1].find(word)
    #     if ind != -1:
    #         current_end = ind
    #         for j in range(i):
    #             current_end -= len(answer_words[j]) + 1
    #         if fuzz.ratio(answer, window[start : len(answer)-current_end-1]) > best_ratio:
    #             best_ratio = fuzz.ratio(answer, window[start : len(answer)-current_end-1])
    #             end = current_end
    # return start, len(answer)-end-1
    return start, start + int(len(answer)*1.1)

def search_phrase(phrase, csv_data, qNumber):
    start_index = concatenated_string.index(phrase)
    end_index = start_index + len(phrase)
    corresponding_indices = []
    len_sum = 0
    for idx, row in csv_data.iterrows():
        text = row["text"]
        len_sum += len(text)+1
            
        if start_index < len_sum:
            corresponding_indices.append(idx)
            csv_data.loc[csv_data.index==idx, 'tagged_sequence'] = f"a__q{qNumber+1}"
            csv_data.loc[csv_data.index==idx, 'highlighted_xpaths'] = csv_data.loc[csv_data.index==idx, 'xpaths']
            csv_data.loc[csv_data.index==idx, 'highlighted_segmented_text'] = csv_data.loc[csv_data.index==idx, 'text']
            if end_index <= len_sum:
                break
    return csv_data
    
# %%
stt = time.time()
contract_idx = 0
nw=[]
json_file_path = 'maud_squad_train.json'
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)
contract_num = json_data["data"][contract_idx]['title']
csv_data = pd.read_csv(f'./contracts/train/{contract_num}.csv')
concatenated_string = " ".join(csv_data["text"])
concatenated_string = concatenated_string.replace("\xa0", " ")
print(contract_num)
for qNumber in range(10, 11):
    print("Q.No ", qNumber)
    target_contract_question = json_data['data'][contract_idx]['paragraphs'][0]['qas'][qNumber]
    contract_num = json_data["data"][contract_idx]['title']
    if not target_contract_question["is_impossible"]:
        answers = target_contract_question["answers"]
        for j in range(len(answers)):
            target_string = answers[j]["text"]
            lrs, rrs, start, window = binary_search_fuzzy(concatenated_string, target_string)
            
            target_string_words = target_string.split(" ")
            cut_res = cut_window(window, target_string, target_string_words)
            if cut_res is None:
                print(f"\t\t Q.No {qNumber}, A.No {j} Trying Linear")
                window = linear_search_fuzzy(concatenated_string, target_string)
                cut_res = cut_window(window, target_string, target_string_words)
                
                big_stride = len(concatenated_string)//15
                small_stride = len(target_string)//5
                
                while cut_res is None:
                    print('\t\t Trying smaller stride')
                    big_stride = big_stride//2
                    small_stride = small_stride//2
                    if small_stride == 1:
                        cut_res = [0,len(window)-1]
                        break
                    window = linear_search_fuzzy(concatenated_string, target_string, big_stride, small_stride)
                    cut_res = cut_window(window, target_string, target_string_words)
                if cut_res is None:
                    nw.append(f"Q.No {qNumber}, A.No {j}")
                if cut_res is not None:
                    print('\t\t Successful with Linear')
            start = cut_res[0]
            end = cut_res[1]
            window_subsection = window[start:end]
            print(f"The answer is: \n {target_string} \n The Subsection \n {window_subsection} \n The Window is \n {window}")
            csv_data = search_phrase(window_subsection, csv_data, qNumber)
            print(f"\t A.No {j} Done ")
ett = time.time()
print(f"time for contract {contract_num} is {ett-stt}")
print("Unsucessful answers")
print(nw)
# %%
csv_data.to_csv(f'{contract_num}.csv',index=False, index_label=None)
# %%

#Contract_idx - not found in directory - [3]


# %%
