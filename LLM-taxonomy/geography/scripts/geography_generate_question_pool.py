import csv
import random
from tqdm import tqdm

child2parent_dict = {} #should be record by code
parent2child_dict = {}
root_set = []

## get root code
with open('./rootcode.txt', 'r') as file:
    content = file.readlines()

root_code_dict = {}

for line in content:
    row_content = line.strip().split('|')
    code = row_content[0].strip()
    name = row_content[1].strip()
    root_code_dict[code] = name

root_set = list(root_code_dict.keys())

## get parent code
with open('./featureCodes_en.txt', 'r') as file:
    content = file.readlines()

level_1_code_dict = {}

for line in content:
    row_content = line.strip().split('\t')
    level_1_code = row_content[0]
    level_1_name = row_content[1]
    root_code = level_1_code.split('.')[0]
    current_code = level_1_code.split('.')[1]
    level_1_code_dict[current_code] = level_1_name
    child2parent_dict[current_code] = [root_code]
    if not root_code in parent2child_dict.keys():
        parent2child_dict[root_code] = [current_code]
    else:
        parent2child_dict[root_code].append(current_code)


code2name_dict = {}
code2name_dict.update(root_code_dict)
code2name_dict.update(level_1_code_dict)

print(len(root_set))

level_1 = []
for item in root_set:
    if item in parent2child_dict.keys():
        level_1+=parent2child_dict[item]

print(len(list(set(level_1))))


level_nodes = [root_set, level_1]

def get_cur_level_uncles(cur_level_nodes, level_nodes, level):
    uncle_dict = {}
    if level == 1:
        for node in cur_level_nodes:
            uncle_dict[node] = list(set(level_nodes[0])-set(child2parent_dict[node]))
        return uncle_dict
    else:
        for node in cur_level_nodes:
            cur_parent_node = child2parent_dict[node][0]
            cur_grand_node = child2parent_dict[cur_parent_node][0]
            cur_uncles = parent2child_dict[cur_grand_node]
            if len(cur_uncles) > len(child2parent_dict[node]):
                uncle_dict[node] = list(set(cur_uncles)-set(child2parent_dict[node]))
            else:
                uncle_dict[node] = list(set(level_nodes[level-1])-set(child2parent_dict[node]))
        return uncle_dict

def sample_question_pool(cur_level_nodes, level_nodes, nodes_uncles_dict, sample_size, cur_level, out_path, question_mode=0):
    random.seed(20)
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 0:
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            for node in tqdm(sampled_nodes):
                if cur_level < 2:
                    cur_template = ('geography-geonames', code2name_dict[child2parent_dict[node][0]], code2name_dict[node], cur_level, 'level-positive')    
                else:
                    cur_template = ('geography-geonames', code2name_dict[child2parent_dict[node][0]], node, cur_level, 'level-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_nodes), cur_level)
        elif question_mode == 1: # negative hard question
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            
            for node in tqdm(sampled_nodes):
                if cur_level < 2:
                    cur_template = ('geography-geonames', code2name_dict[random.choice(nodes_uncles_dict[node])], code2name_dict[node], cur_level, 'level-negative-hard')    
                else:
                    cur_template = ('geography-geonames', code2name_dict[random.choice(nodes_uncles_dict[node])], node, cur_level, 'level-negative-hard')
                csv_writer.writerow(cur_template)
            print('size of the negative set', len(sampled_nodes), cur_level)

        elif question_mode == 2:
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            
            for node in tqdm(sampled_nodes):
                if cur_level < 2:
                    cur_template = ('geography-geonames', code2name_dict[random.choice(list(set(level_nodes[cur_level-1])-set(child2parent_dict[node])))], code2name_dict[node], cur_level, 'level-negative-easy')    
                else:
                    cur_template = ('geography-geonames', code2name_dict[random.choice(list(set(level_nodes[cur_level-1])-set(child2parent_dict[node])))], node, cur_level, 'level-negative-easy')
                csv_writer.writerow(cur_template)
            print('size of the negative set', len(sampled_nodes), cur_level)


num_question_samples = [100000]

file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv']


for cur_question_mode in range(3):
    out_path = './TaxoGlimpse/question_pools/geography-geonames/level/level_question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    
    for cur_level in range(1, len(num_question_samples)+1):
        nodes_uncles_dict = get_cur_level_uncles(level_nodes[cur_level], level_nodes, cur_level) # level 1是从level 1 到root
        sample_question_pool(level_nodes[cur_level], level_nodes, nodes_uncles_dict, sample_size=num_question_samples[cur_level-1], cur_level=cur_level, out_path=out_path, question_mode=cur_question_mode)

