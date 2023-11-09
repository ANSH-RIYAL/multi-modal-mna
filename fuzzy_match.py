# %%
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm
import time
import json
import re

# %%
csv_data = pd.read_csv('.\\contracts\\train\\contract_60.csv')


# %%
concatenated_string = " ".join(csv_data["text"])
concatenated_string = concatenated_string.replace("\xa0", " ")

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
def binary_search_fuzzy(context, answer):
    start = 0
    end = len(context)
    
    thres = 90
    
    left_ratios = []
    right_ratios = []
    
    while (end - start > 1.6*len(answer)):
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


# %%
def cut_window(window, answer, answer_words):
    thres = 70
    answer_words = answer.split()
    for i in range(len(answer_words)):
        word = answer_words[i]
        ind = window.find(word)
        if ind != -1:
            start = ind
            for j in range(i):
                start -= len(answer_words[j]) + 1
            if fuzz.partial_ratio(answer, window[start:start + len(answer)]) > thres:
                return (start, start + len(answer))

# %%
def search_phrase(phrase, csv_data):
    start_index = concatenated_string.index(phrase)
    end_index = start_index + len(phrase)
    corresponding_indices = []
    len_sum = 0
    for idx, row in csv_data.iterrows():
        text = row["text"]
        len_sum += len(text)+1
            
        if start_index < len_sum:
            corresponding_indices.append(idx)
            csv_data.loc[csv_data.index==idx, 'tagged_sequence'] = 'b_y'
            csv_data.loc[csv_data.index==idx, 'highlighted_xpaths'] = csv_data.loc[csv_data.index==idx, 'xpaths']
            csv_data.loc[csv_data.index==idx, 'highlighted_segmented_text'] = csv_data.loc[csv_data.index==idx, 'text']
            if end_index <= len_sum:
                break
    return csv_data
    
# %%
contract = 0
for qNumber in range(0, 22):
    print(qNumber)
    target_contract_question = json_data['data'][contract]['paragraphs'][0]['qas'][qNumber]
    contract_num = json_data["data"][contract]['title']
    csv_data = pd.read_csv(f'.\\contracts\\train\\{contract_num}.csv')
    if not target_contract_question["is_impossible"]:
        answers = target_contract_question["answers"]
        for j in range(len(answers)):
            target_string = answers[j]["text"]
            print(j)
            lrs, rrs, start, window = binary_search_fuzzy(concatenated_string, target_string)
            target_string_words = target_string.split(" ")
            cut_res = cut_window(window, target_string, target_string_words)
            if cut_res is None:
                print(f"Question {qNumber}, Answer {j} Unsuccessful")
            else:
                start = cut_res[0]
                end = cut_res[1]
                window_subsection = window[start:end]
                csv_data = search_phrase(window_subsection, csv_data)

# %%
qNumber = 6
target_contract_question = json_data['data'][contract]['paragraphs'][0]['qas'][qNumber]
contract_num = json_data["data"][contract]['title']
csv_data = pd.read_csv(f'.\\contracts\\train\\{contract_num}.csv')
if not target_contract_question["is_impossible"]:
    answers = target_contract_question["answers"]
    j = 0
    target_string = answers[j]["text"]
    print(target_string)
    lrs, rrs, start, window = binary_search_fuzzy(concatenated_string, target_string)
    print(window)
    target_string_words = target_string.split(" ")
    cut_res = cut_window(window, target_string, target_string_words)
    if cut_res is None:
        print(f"Question {qNumber}, Answer {j} Unsuccessful")
    else:
        start = cut_res[0]
        end = cut_res[1]
        window_subsection = window[start:end]
        csv_data = search_phrase(window_subsection, csv_data)
# %%
csv_data.to_csv('very_new_csv.csv',index=False, index_label=None)
# %%
