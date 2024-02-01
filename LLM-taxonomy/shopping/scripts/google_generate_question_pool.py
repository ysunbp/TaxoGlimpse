import os
import csv
import random
from tqdm import tqdm
import time


def process_txt_files(txt_path, level): # level这里从0开始数
    cur_level_nodes = []
    cur_level_nodes_str = []
    nodes_children_dict = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            else:
                nodes = line.split('>')
                updated_nodes = []
                for node in nodes:
                    updated_nodes.append(node.strip())
                if level < 3:
                    if len(updated_nodes) < level+2:
                        continue
                    elif level == 0:
                        cur_pair = (updated_nodes[level], updated_nodes[level+1], 'root')
                        if not str(cur_pair) in cur_level_nodes_str:
                            cur_level_nodes_str.append(str(cur_pair))
                            cur_level_nodes.append(cur_pair)
                            if not cur_pair[2] in nodes_children_dict.keys():
                                nodes_children_dict[cur_pair[2]] = [cur_pair[0]]
                            else:
                                nodes_children_dict[cur_pair[2]].append(cur_pair[0])

                    else:
                        cur_pair = (updated_nodes[level], updated_nodes[level+1], updated_nodes[level-1])
                        if not str(cur_pair) in cur_level_nodes_str:
                            cur_level_nodes_str.append(str(cur_pair))
                            cur_level_nodes.append(cur_pair)
                            if not cur_pair[2] in nodes_children_dict.keys():
                                nodes_children_dict[cur_pair[2]] = [cur_pair[0]]
                            else:
                                nodes_children_dict[cur_pair[2]].append(cur_pair[0])
                else:
                    if len(updated_nodes) < level+2:
                        continue
                    else:
                        for node_idx in range(level+1, len(updated_nodes)):
                            cur_pair = (updated_nodes[node_idx-1], updated_nodes[node_idx], updated_nodes[node_idx-2])
                            if not str(cur_pair) in cur_level_nodes_str:
                                cur_level_nodes_str.append(str(cur_pair))
                                cur_level_nodes.append(cur_pair)
                                if not cur_pair[2] in nodes_children_dict.keys():
                                    nodes_children_dict[cur_pair[2]] = [cur_pair[0]]
                                else:
                                    nodes_children_dict[cur_pair[2]].append(cur_pair[0])    
    print('total number of pairs', len(cur_level_nodes))
    return cur_level_nodes, nodes_children_dict

def sample_question_pool(cur_level_pairs, nodes_children_dict, cur_level, out_path, sample_size, question_mode = 0):
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 1: # negative question hard
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


            for cur_idx, sampled_pair in enumerate(sampled_pairs):
                cur_root_set = list(set(nodes_children_dict[sampled_pair[2]])-set([sampled_pair[0]]))
                if len(cur_root_set)==0:
                    #print('haha', cur_idx)
                    #cur_root_set = list(set(root_set)-set([sampled_pair[0]]))
                    continue
                negative_sampled_pairs.append((random.choice(cur_root_set), sampled_pair[1]))

            for idx, (root, child) in enumerate(tqdm(negative_sampled_pairs)):
                cur_template = ('shopping-google', root, child, cur_level, 'level-negative-hard')
                csv_writer.writerow(cur_template)
            
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        elif question_mode == 0:
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            
            for (root, child, _) in tqdm(sampled_pairs):
                cur_template = ('shopping-google', root, child, cur_level, 'level-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_pairs), cur_level)
        elif question_mode == 2: # negative-easy
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
                cur_template = ('shopping-google', root, child, cur_level, 'level-negative-easy')
                csv_writer.writerow(cur_template)
            
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        elif question_mode == 3:
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            
            for (root, child, _) in tqdm(sampled_pairs):
                cur_template = ('shopping-google', root, child, cur_level, 'toroot-positive')
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
                cur_template = ('shopping-google', root, child, cur_level, 'toroot-negative-easy')
                csv_writer.writerow(cur_template)
            
            print('size of the negative set', len(negative_sampled_pairs), cur_level)

def process_txt_files_to_root(txt_path, level): # level这里从0开始数
    cur_level_nodes = []
    cur_level_nodes_str = []
    nodes_children_dict = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx == 0:
                continue
            else:
                nodes = line.split('>')
                updated_nodes = []
                for node in nodes:
                    updated_nodes.append(node.strip())
                if level < 3:
                    if len(updated_nodes) < level+2:
                        continue
                    elif level == 0:
                        cur_pair = (updated_nodes[0], updated_nodes[level+1], 'root')
                        if not str(cur_pair) in cur_level_nodes_str:
                            cur_level_nodes_str.append(str(cur_pair))
                            cur_level_nodes.append(cur_pair)
                            if not cur_pair[2] in nodes_children_dict.keys():
                                nodes_children_dict[cur_pair[2]] = [cur_pair[0]]
                            else:
                                nodes_children_dict[cur_pair[2]].append(cur_pair[0])

                    else:
                        cur_pair = (updated_nodes[0], updated_nodes[level+1], updated_nodes[level-1])
                        if not str(cur_pair) in cur_level_nodes_str:
                            cur_level_nodes_str.append(str(cur_pair))
                            cur_level_nodes.append(cur_pair)
                            if not cur_pair[2] in nodes_children_dict.keys():
                                nodes_children_dict[cur_pair[2]] = [cur_pair[0]]
                            else:
                                nodes_children_dict[cur_pair[2]].append(cur_pair[0])
                else:
                    if len(updated_nodes) < level+2:
                        continue
                    else:
                        for node_idx in range(level+1, len(updated_nodes)):
                            cur_pair = (updated_nodes[0], updated_nodes[node_idx], updated_nodes[node_idx-2])
                            if not str(cur_pair) in cur_level_nodes_str:
                                cur_level_nodes_str.append(str(cur_pair))
                                cur_level_nodes.append(cur_pair)
                                if not cur_pair[2] in nodes_children_dict.keys():
                                    nodes_children_dict[cur_pair[2]] = [cur_pair[0]]
                                else:
                                    nodes_children_dict[cur_pair[2]].append(cur_pair[0])    
    print('total number of pairs', len(cur_level_nodes))
    return cur_level_nodes, nodes_children_dict

def setup_seed(seed):
    random.seed(seed)

txt_path = 'TaxoGlimpse/LLM-taxonomy/shopping/data/google-US.txt'

num_question_samples = [100000, 100000, 100000, 100000]

file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv', 'positive_to_root.csv', 'negative_to_root.csv']

for cur_question_mode in range(5):
    if cur_question_mode < 3:
        out_path = 'TaxoGlimpse/question_pools/shopping-google/level/level_question_pool_full_' + file_name[cur_question_mode]
        with open(out_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
        for cur_level in range(len(num_question_samples)):
            setup_seed(20)
            cur_level_pairs, nodes_children_dict = process_txt_files(txt_path, level=cur_level)
            sample_question_pool(cur_level_pairs, nodes_children_dict, cur_level+1, out_path, sample_size=num_question_samples[cur_level], question_mode = cur_question_mode) # question mode 0 is positive, question mode 1 is negative-hard, question mode 2 is negative-easy
    else:
        out_path = 'TaxoGlimpse/question_pools/shopping-google/toroot/level_question_pool_full_' + file_name[cur_question_mode]
        with open(out_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
        for cur_level in range(len(num_question_samples)):
            setup_seed(20)
            cur_level_pairs, nodes_children_dict = process_txt_files_to_root(txt_path, level=cur_level)
            sample_question_pool(cur_level_pairs, nodes_children_dict, cur_level+1, out_path, sample_size=num_question_samples[cur_level], question_mode = 4) # question mode 3 is positive-toroot, question mode 4 is negative-toroot
