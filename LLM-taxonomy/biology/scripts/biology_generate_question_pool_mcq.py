import os
import csv
import random
from tqdm import tqdm
import time
import pickle


def process_name_dict(dump_path):
    name_dict = {}
    with open(dump_path, 'r') as f:
        Lines = f.readlines()
    for line in tqdm(Lines):
        line = line.strip()
        striped_elements = line.split('|')
        tax_id = striped_elements[0].strip()
        name_type = striped_elements[3].strip()
        name = striped_elements[1].strip()
        if (name_type == 'scientific name') and (not tax_id in name_dict.keys()):
            name_dict[tax_id] = name
    return name_dict

def sample_nodes_hard(cur_level_pairs, sample_size, name_dict, node_map_to_parent, parent_map_to_nodes, out_path, cur_level, question_mode = 0, base_set=None):
    level = cur_level
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)

        if question_mode == 0: # MCQ hard
            root_set = []
            for pair in tqdm(cur_level_pairs):
                if pair[0] in root_set:
                    continue
                else:
                    root_set.append(pair[0])

            if level > 1:
                uncle_dict = parent_map_to_nodes
            
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            for idx, sampled_pair in enumerate(sampled_pairs):
                if level > 1:
                    cur_uncles = uncle_dict[node_map_to_parent[sampled_pair[0]]]
                    root_set = cur_uncles
                cur_root_set = list(set(root_set)-set([sampled_pair[0]]))
                if len(cur_root_set) < 3:
                    sampled_indices = random.sample(list(set(list(range(len(sampled_pairs))))-set([idx])), 3-len(cur_root_set))
                    sampled_complements = [sampled_pairs[sampled_idx][1] for sampled_idx in sampled_indices]
                    selected_mcqs = cur_root_set + sampled_complements
                else:
                    selected_mcqs = random.sample(cur_root_set, 3)
                
                root, child = sampled_pair[0], sampled_pair[1]
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('biology-ncbi', root_name, child_name, cur_level, 'mcq-negative-hard', name_dict[selected_mcqs[0]], name_dict[selected_mcqs[1]], name_dict[selected_mcqs[2]])
                csv_writer.writerow(cur_template)
            print('size of the MCQ set', len(sampled_pairs), cur_level)

        elif question_mode == 1:
            root_set = []
            for pair in tqdm(cur_level_pairs):
                if pair[0] in root_set:
                    continue
                else:
                    root_set.append(pair[0])
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            cur_root_set = list(set(root_set))
            cur_root_choices = random.choices(cur_root_set, k=len(sampled_pairs)*3)
            for idx, sampled_pair in enumerate(tqdm(sampled_pairs)):
                cur_root_choice_a = cur_root_choices[idx]
                cur_root_choice_b = cur_root_choices[idx+len(sampled_pairs)]
                cur_root_choice_c = cur_root_choices[idx+len(sampled_pairs)*2]
                while cur_root_choice_a == sampled_pair[0]:
                    cur_root_choice_a = random.choice(cur_root_set)
                while cur_root_choice_b == cur_root_choice_a or cur_root_choice_b == sampled_pair[0]:
                    cur_root_choice_b = random.choice(cur_root_set)
                while cur_root_choice_c == cur_root_choice_b or cur_root_choice_c == cur_root_choice_a or cur_root_choice_c == sampled_pair[0]:
                    cur_root_choice_c = random.choice(cur_root_set)
                selected_mcqs = [cur_root_choice_a, cur_root_choice_b, cur_root_choice_c]
                root = sampled_pair[0]
                child = sampled_pair[1]
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('biology-ncbi', root_name, child_name, cur_level, 'mcq-negative-easy', name_dict[selected_mcqs[0]], name_dict[selected_mcqs[1]], name_dict[selected_mcqs[2]])
                csv_writer.writerow(cur_template)
                    
            print('size of the MCQ set', len(sampled_pairs), cur_level)


def setup_seed(seed):
    random.seed(seed)

setup_seed(20)

num_question_samples = [100000000, 100000000, 100000000, 100000000, 100000000, 100000000]

file_name = ['mcq_hard.csv', 'mcq_easy.csv']

name_dict_path = 'TaxoGlimpse/LLM-taxonomy/biology/NBCI/taxdmp/names.dmp'
name_dict = process_name_dict(name_dict_path)

for idx in range(6):
    cur_level = idx+1
    with open('TaxoGlimpse/LLM-taxonomy/biology/temp/level_'+str(idx+1)+'.pkl', 'rb') as pickle_file:
        cur_level_pairs = pickle.load(pickle_file)
    if idx > 0:
        with open('TaxoGlimpse/LLM-taxonomy/biology/temp/level_'+str(idx)+'.pkl', 'rb') as pickle_file:
            higher_level_pairs = pickle.load(pickle_file)
    else:
        higher_level_pairs = cur_level_pairs

    with open('TaxoGlimpse/LLM-taxonomy/biology/temp/node2parent.pkl', 'rb') as pickle_file:
        node_map_to_parent = pickle.load(pickle_file)
    with open('TaxoGlimpse/LLM-taxonomy/biology/temp/parent2nodes.pkl', 'rb') as pickle_file:
        parent_map_to_nodes = pickle.load(pickle_file)
    
    for cur_question_mode in range(2):
        out_path = 'TaxoGlimpse/question_pools/biology-NCBI/mcq/question_pool_full_level_' + str(cur_level) + '_' + file_name[cur_question_mode]
        with open(out_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type', 'mcq-parent1', 'mcq-parent2', 'mcq-parent3'])  # 写入表头
        sample_nodes_hard(cur_level_pairs, sample_size=num_question_samples[idx], out_path=out_path, cur_level=cur_level, name_dict=name_dict, node_map_to_parent = node_map_to_parent, parent_map_to_nodes=parent_map_to_nodes, question_mode=cur_question_mode)
    
