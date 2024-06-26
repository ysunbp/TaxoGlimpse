import os
import numpy as np

data_path = 'TaxoGlimpse/LLM-taxonomy/academic/data/acm_ccs2012-1626988337597.xml'

import csv
import random
from tqdm import tqdm
import time
import pickle


def process_xml(data_path):
    with open(data_path, 'r') as f:
        Lines = f.readlines()
    id_list = []
    name_list = []
    for line in Lines:
        line = line.strip()
        if line.split(' ')[0] == '<skos:Concept':
            concept_id = line.split(' ')[1][:-1].split('=')[1][1:-1]
            id_list.append(concept_id)
        elif line.split(' ')[0] == '<skos:prefLabel':
            concept_name = line.split('>')[1].split('<')[0]
            name_list.append(concept_name)
    dictionary = {k: v for k, v in zip(id_list, name_list)}
    return id_list, name_list, dictionary

def process_level(id_list, cur_level):
    cur_level_pairs = []
    cur_level_pairs_str = []
    for concept_id in id_list:
        concept_chain = concept_id.split('.')
        if len(concept_chain) < cur_level+1:
            continue
        else:
            cur_sub_id = concept_chain[0]
            for idx in range(1, cur_level+1):
                cur_sub_id  = cur_sub_id + '.'
                cur_sub_id += concept_chain[idx]
            cur_root_id = concept_chain[0]
            if cur_level > 1:
                for idx in range(1, cur_level):
                    cur_root_id = cur_root_id + '.'
                    cur_root_id += concept_chain[idx]
            if not str((cur_root_id, cur_sub_id)) in cur_level_pairs_str:
                cur_level_pairs.append((cur_root_id, cur_sub_id))
                cur_level_pairs_str.append(str((cur_root_id, cur_sub_id)))
    return cur_level_pairs

def get_uncles(cur_level_pairs, cur_level):
    
    cur_uncles = []
    all_parents = []
    for cur_pair in cur_level_pairs:
        cur_parent = cur_pair[0]
        if not cur_parent in all_parents:
            all_parents.append(cur_parent)
    
    for idx, cur_pair in enumerate(cur_level_pairs):
        if cur_level == 1:
            cur_uncles.append(all_parents)
        else:
            cur_grand = cur_pair[1].split('.')[-3]
            cur_uncles.append([])
            for cur_candidate in all_parents:
                if cur_candidate.split('.')[-2] == cur_grand:
                    cur_uncles[idx].append(cur_candidate)
    return cur_uncles

def setup_seed(seed):
    random.seed(seed)

def sample_question_pool(cur_level_pairs, sample_size, name_dict, uncles_list, out_path, cur_level, question_mode = 0):
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)

        if question_mode == 0: # MCQ hard
            root_set = []
            for pair in cur_level_pairs:
                if pair[0] in root_set:
                    continue
                else:
                    root_set.append(pair[0])
            if len(cur_level_pairs) < sample_size:
                sampled_pairs_idx = range(len(cur_level_pairs))
            else:
                sampled_pairs_idx = random.sample(range(len(cur_level_pairs)), sample_size)
            for cur_idx, sampled_pair_idx in enumerate(sampled_pairs_idx):
                sampled_pair = cur_level_pairs[sampled_pair_idx]
                cur_uncles = uncles_list[sampled_pair_idx]
                
                root, child = sampled_pair[0], sampled_pair[1]
                root_name = name_dict[root]
                child_name = name_dict[child]

                if len(cur_uncles) < 4: #uncle里面有自己的parent
                    cur_root_set = list(set(cur_uncles)-set([sampled_pair[0]]))
                    sampled_indices = random.sample(list(set(list(range(len(sampled_pairs_idx))))-set([cur_idx])), 3-len(cur_root_set))
                    sampled_complements = [cur_level_pairs[sampled_idx][1] for sampled_idx in sampled_indices]
                    selected_mcqs = cur_root_set + sampled_complements
                    
                else:
                    cur_root_set = list(set(cur_uncles)-set([sampled_pair[0]]))
                    selected_mcqs = random.sample(cur_root_set, 3) 
                cur_template = ('academic-acm', root_name, child_name, cur_level, 'mcq-negative-hard', name_dict[selected_mcqs[0]], name_dict[selected_mcqs[1]], name_dict[selected_mcqs[2]])
                csv_writer.writerow(cur_template)
                
            print('size of the MCQ set', len(sampled_pairs_idx), cur_level)
        
        elif question_mode == 1:
            root_set = []
            for pair in cur_level_pairs:
                if pair[0] in root_set:
                    continue
                else:
                    root_set.append(pair[0])
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            for sampled_pair in sampled_pairs:
                cur_root_set = list(set(root_set)-set([sampled_pair[0]])) #此处不考虑size因为必须大于3
                selected_mcqs = random.sample(cur_root_set, 3) 
                root, child = sampled_pair[0], sampled_pair[1]
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('academic-acm', root_name, child_name, cur_level, 'mcq-negative-easy', name_dict[selected_mcqs[0]], name_dict[selected_mcqs[1]], name_dict[selected_mcqs[2]])
                csv_writer.writerow(cur_template)
                
            print('size of the MCQ set', len(sampled_pairs), cur_level)
        
        
setup_seed(20)
id_list, name_list, dictionary = process_xml(data_path)

num_question_samples = [100000, 100000, 100000, 100000]
file_name = ['mcq_hard.csv', 'mcq_easy.csv']


for cur_question_mode in range(2):
    out_path = 'TaxoGlimpse/question_pools/academic-acm/mcq/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type', 'mcq-parent1', 'mcq-parent2', 'mcq-parent3'])  # 写入表头
    for cur_level in range(1, len(num_question_samples)+1):
        if cur_level < 4:
            cur_level_pairs = process_level(id_list, cur_level=cur_level) # level从1开始数 level=1即从level 1 to root; 4和5的要单独处理再合并
            cur_uncles = get_uncles(cur_level_pairs, cur_level=cur_level)
            sample_question_pool(cur_level_pairs, num_question_samples[cur_level-1], dictionary, cur_uncles, out_path, cur_level, question_mode = cur_question_mode)
        else:
            cur_level_pairs_a = process_level(id_list, cur_level=cur_level) # level从1开始数 level=1即从level 1 to root; 4和5的要单独处理再合并
            cur_uncles_a = get_uncles(cur_level_pairs_a, cur_level=cur_level)
            cur_level_pairs_b = process_level(id_list, cur_level=cur_level+1) # level从1开始数 level=1即从level 1 to root; 4和5的要单独处理再合并
            cur_uncles_b = get_uncles(cur_level_pairs_b, cur_level=cur_level+1)
            cur_level_pairs = cur_level_pairs_a + cur_level_pairs_b
            cur_uncles = cur_uncles_a + cur_uncles_b
            sample_question_pool(cur_level_pairs, num_question_samples[cur_level-1], dictionary, cur_uncles, out_path, cur_level, question_mode = cur_question_mode)
