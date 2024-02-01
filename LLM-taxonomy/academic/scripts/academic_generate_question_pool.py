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

def process_level_root(id_list, cur_level):
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
                cur_level_pairs.append((cur_sub_id.split('.')[0], cur_sub_id))
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
        if question_mode == 1: # negative
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
            negative_sampled_pairs = []
            for cur_idx, sampled_pair_idx in enumerate(sampled_pairs_idx):
                sampled_pair = cur_level_pairs[sampled_pair_idx]
                cur_uncles = uncles_list[sampled_pair_idx]
                if len(cur_uncles) < 2:
                    cur_root_set = list(set(root_set)-set([sampled_pair[0]]))
                    continue
                else:
                    cur_root_set = list(set(cur_uncles)-set([sampled_pair[0]]))
                negative_sampled_pairs.append((random.choice(cur_root_set),sampled_pair[1]))
            for (root, child) in tqdm(negative_sampled_pairs):
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('academic-acm', root_name, child_name, cur_level, 'level-negative-hard')
                csv_writer.writerow(cur_template)
                
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        
        elif question_mode == 0:
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            for (root, child) in tqdm(sampled_pairs):
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('academic-acm', root_name, child_name, cur_level, 'level-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_pairs), cur_level)

        elif question_mode == 2:
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
            negative_sampled_pairs = []
            for sampled_pair in sampled_pairs:
                cur_root_set = list(set(root_set)-set([sampled_pair[0]]))
                negative_sampled_pairs.append((random.choice(cur_root_set),sampled_pair[1]))
            for (root, child) in tqdm(negative_sampled_pairs):
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('academic-acm', root_name, child_name, cur_level, 'level-negative-easy')
                csv_writer.writerow(cur_template)
                
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        
        elif question_mode == 3:
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            for (root, child) in tqdm(sampled_pairs):
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('academic-acm', root_name, child_name, cur_level, 'toroot-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_pairs), cur_level)
        elif question_mode == 4:
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
            negative_sampled_pairs = []
            for sampled_pair in sampled_pairs:
                cur_root_set = list(set(root_set)-set([sampled_pair[0]]))
                negative_sampled_pairs.append((random.choice(cur_root_set),sampled_pair[1]))
            for (root, child) in tqdm(negative_sampled_pairs):
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('academic-acm', root_name, child_name, cur_level, 'toroot-negative')
                csv_writer.writerow(cur_template)
                
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        
setup_seed(20)
id_list, name_list, dictionary = process_xml(data_path)

num_question_samples = [100000, 100000, 100000, 100000]
file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv', 'positive_to_root.csv', 'negative_to_root.csv']

for cur_question_mode in range(3):
    out_path = 'TaxoGlimpse/question_pools/academic-acm/level/level_question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
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

for cur_question_mode in range(3,5):
    out_path = 'TaxoGlimpse/question_pools/academic-acm/toroot/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    for cur_level in range(1, len(num_question_samples)+1):
        if cur_level < 4:
            cur_level_pairs = process_level_root(id_list, cur_level=cur_level) # level从1开始数 level=1即从level 1 to root; 4和5的要单独处理再合并
            cur_uncles = None
            sample_question_pool(cur_level_pairs, num_question_samples[cur_level-1], dictionary, cur_uncles, out_path, cur_level, question_mode = cur_question_mode)
        else:
            cur_level_pairs_a = process_level_root(id_list, cur_level=cur_level) # level从1开始数 level=1即从level 1 to root; 4和5的要单独处理再合并
            
            cur_level_pairs_b = process_level_root(id_list, cur_level=cur_level+1) # level从1开始数 level=1即从level 1 to root; 4和5的要单独处理再合并
            
            cur_level_pairs = cur_level_pairs_a + cur_level_pairs_b
            cur_uncles = None
            sample_question_pool(cur_level_pairs, num_question_samples[cur_level-1], dictionary, cur_uncles, out_path, cur_level, question_mode = cur_question_mode)
