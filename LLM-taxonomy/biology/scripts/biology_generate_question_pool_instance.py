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

def process_node_file_to_parent_dict(node_path):
    node_map_to_parent = {}
    parent_map_to_nodes = {}
    with open(node_path, 'r') as f:
        Lines = f.readlines()
    for line in tqdm(Lines):
        line = line.strip()
        striped_elements = line.split('|')
        tax_id = striped_elements[0].strip()
        parent_id = striped_elements[1].strip()
        node_map_to_parent[tax_id] = parent_id
        if parent_id in parent_map_to_nodes.keys():
            parent_map_to_nodes[parent_id].append(tax_id)
        else:
            parent_map_to_nodes[parent_id] = [tax_id]
    return node_map_to_parent, parent_map_to_nodes

def process_node_file(node_path, level, levels):
    cur_level_pairs = []
    cur_level_pairs_str = []
    with open(node_path, 'r') as f:
        Lines = f.readlines()
    for line in tqdm(Lines):
        line = line.strip()
        striped_elements = line.split('|')
        tax_id = striped_elements[0].strip()
        parent_id = striped_elements[1].strip()
        category = striped_elements[2].strip()
        cur_sub_cat = levels[level]
        cur_root_cat = levels[level-1]
        if category == cur_sub_cat:
            if not str((parent_id, tax_id)) in cur_level_pairs_str:
                cur_level_pairs.append((parent_id, tax_id))
                cur_level_pairs_str.append(str((parent_id, tax_id)))
    return cur_level_pairs

def get_base_root_set(level_1_path='TaxoGlimpse/LLM-taxonomy/biology/temp/level_1.pkl'):
    with open(level_1_path, 'rb') as pickle_file:
        cur_level_pairs = pickle.load(pickle_file)
    root_set = []
    for pair in cur_level_pairs:
        if not pair[0] in root_set:
            root_set.append(pair[0])
    return root_set

def get_node_root(node_map_to_parent, cur_pair, base_set):
    cur_node = cur_pair[1]

    while cur_node in node_map_to_parent.keys():
        if cur_node in base_set:
            break
        elif node_map_to_parent[cur_node] == '1':
            cur_node = None
            break
        cur_node = node_map_to_parent[cur_node]
    return cur_node


def sample_nodes_hard_instance(cur_level_pairs, sample_size, name_dict, node_map_to_parent, parent_map_to_nodes, out_path, cur_level, question_mode = 0, base_set=None):
    level = cur_level
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 1: #negative hard
            root_set = []
            for pair in tqdm(cur_level_pairs):
                if pair[0] in root_set:
                    continue
                else:
                    root_set.append(pair[0])

            if level > 0:
                uncle_dict = parent_map_to_nodes
            
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            negative_sampled_pairs = []
            for idx, sampled_pair in enumerate(sampled_pairs):

                if level > 0:
                    cur_uncles = uncle_dict[node_map_to_parent[sampled_pair[0]]]
                    if len(cur_uncles) > 1:
                        root_set = cur_uncles
                    else:
                        continue
                
                cur_root_set = list(set(root_set)-set([sampled_pair[0]]))
                negative_sampled_pairs.append((random.choice(cur_root_set),sampled_pair[1]))
            for (root, child) in tqdm(negative_sampled_pairs):
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('biology-ncbi', root_name, child_name, 'instance'+str(cur_level), 'level-negative-hard')
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
                cur_template = ('biology-ncbi', root_name, child_name, 'instance'+str(cur_level), 'level-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_pairs), cur_level)

        elif question_mode == 2:
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
            negative_sampled_pairs = []
            cur_root_set = list(set(root_set))
            cur_root_choices = random.choices(cur_root_set, k=len(sampled_pairs))
            for idx, sampled_pair in enumerate(tqdm(sampled_pairs)):
                cur_root_choice = cur_root_choices[idx]
                while cur_root_choice == sampled_pair[0]:
                    cur_root_choice = random.choice(cur_root_set)
                negative_sampled_pairs.append((cur_root_choice, sampled_pair[1]))
            for (root, child) in tqdm(negative_sampled_pairs):
                root_name = name_dict[root]
                child_name = name_dict[child]
                cur_template = ('biology-ncbi', root_name, child_name, 'instance'+str(cur_level), 'level-negative-easy')
                csv_writer.writerow(cur_template)
                    
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
########################################################
########################################################
########################################################

node_path = 'TaxoGlimpse/LLM-taxonomy/biology/NBCI/taxdmp/nodes.dmp'
node_map_to_parent, parent_map_to_nodes = process_node_file_to_parent_dict(node_path)
with open('TaxoGlimpse/LLM-taxonomy/biology/temp/node2parent.pkl', 'wb') as pickle_file:
    pickle.dump(node_map_to_parent, pickle_file)
with open('TaxoGlimpse/LLM-taxonomy/biology/temp/parent2nodes.pkl', 'wb') as pickle_file:
    pickle.dump(parent_map_to_nodes, pickle_file)
levels = ['superkingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

name_dict_path = 'TaxoGlimpse/LLM-taxonomy/biology/NBCI/taxdmp/names.dmp'
node_path = 'TaxoGlimpse/LLM-taxonomy/biology/NBCI/taxdmp/nodes.dmp'

#name_dict = process_name_dict(name_dict_path)
#print(name_dict)

for idx in range(6):
    cur_level_pairs = process_node_file(node_path, idx+1, levels)
    with open('TaxoGlimpse/LLM-taxonomy/biology/temp/level_'+str(idx+1)+'.pkl', 'wb') as pickle_file:
        pickle.dump(cur_level_pairs, pickle_file)

    print(len(cur_level_pairs))
########################################################
########################################################
########################################################

def setup_seed(seed):
    random.seed(seed)

setup_seed(20)

num_question_samples = [100000000, 100000000, 100000000, 100000000, 100000000, 100000000]

file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv', 'positive_to_root.csv', 'negative_to_root.csv']

name_dict_path = 'TaxoGlimpse/LLM-taxonomy/biology/NBCI/taxdmp/names.dmp'
name_dict = process_name_dict(name_dict_path)

with open('TaxoGlimpse/LLM-taxonomy/biology/temp/level_6.pkl', 'rb') as pickle_file:
    instance_pairs = pickle.load(pickle_file)

with open('TaxoGlimpse/LLM-taxonomy/biology/temp/node2parent.pkl', 'rb') as pickle_file:
    node_map_to_parent = pickle.load(pickle_file)
del node_map_to_parent['1']




for idx in range(4,5):
    cur_level = idx+1
    with open('TaxoGlimpse/LLM-taxonomy/biology/temp/level_'+str(idx+1)+'.pkl', 'rb') as pickle_file:
        cur_level_pairs = pickle.load(pickle_file)
    
    cur_level_root_set = []
    for item in cur_level_pairs:
        if item[1] in cur_level_root_set:
            continue
        else:
            cur_level_root_set.append(item[1])
    
    updated_level_pairs = []
    for level_pair in tqdm(instance_pairs):
        cur_node_root = get_node_root(node_map_to_parent, level_pair, cur_level_root_set)
        if cur_node_root:
            updated_level_pairs.append((cur_node_root,level_pair[1]))

    with open('TaxoGlimpse/LLM-taxonomy/biology/temp/parent2nodes.pkl', 'rb') as pickle_file:
        parent_map_to_nodes = pickle.load(pickle_file)
    for cur_question_mode in range(1,3):
        out_path = 'TaxoGlimpse/question_pools/biology-NCBI/instance_full/question_pool_full_' + file_name[cur_question_mode]
        with open(out_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
        sample_nodes_hard_instance(updated_level_pairs, sample_size=num_question_samples[idx], out_path=out_path, cur_level=cur_level, name_dict=name_dict, node_map_to_parent = node_map_to_parent, parent_map_to_nodes=parent_map_to_nodes, question_mode=cur_question_mode, base_set=cur_level_root_set)

