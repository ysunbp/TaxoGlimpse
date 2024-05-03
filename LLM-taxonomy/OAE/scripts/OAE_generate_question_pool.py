import csv
import random
from tqdm import tqdm

id2name_dict = {}
child2parent_dict = {}
parent2child_dict = {}

with open('./OAE.csv', 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)

    # 遍历文件中的每一行
    for idx, row in enumerate(csv_reader):
        if idx == 0:
            continue
        child = row[1]
        child_id = row[0]
        parent_id = row[7]
        #print(child, child_id, parent_id)
        if not child_id in id2name_dict.keys():
            id2name_dict[child_id] = child

with open('./OAE.csv', 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)
    for idx, row in enumerate(csv_reader):
        if idx == 0:
            continue
        child = row[1]
        child_id = row[0]
        parent_id = row[7]
        if not parent_id in id2name_dict.keys():
            continue
        if child == id2name_dict[parent_id]:
            continue
        if not child_id in child2parent_dict.keys():
            child2parent_dict[child_id] = [parent_id]
        else:
            if id2name_dict[parent_id] in [id2name_dict[item] for item in child2parent_dict[child_id]]:
                continue
            child2parent_dict[child_id].append(parent_id)
        if not parent_id in parent2child_dict.keys():
            parent2child_dict[parent_id] = [child_id]
        else:
            parent2child_dict[parent_id].append(child_id)
        

root_set = []
for cur_parent in parent2child_dict.keys():
    if cur_parent in child2parent_dict.keys():
        continue
    else:
        if not cur_parent in root_set:
            root_set.append(cur_parent)

#print(len(root_set))

level_1_code = []
for root_type in root_set:
    if root_type in parent2child_dict.keys():
        level_1_code += parent2child_dict[root_type]

#print(len(level_1_code))

level_2_code = []
for root_type in level_1_code:
    if root_type in parent2child_dict.keys():
        level_2_code += parent2child_dict[root_type]

#print(len(level_2_code))

level_3_code = []
for root_type in level_2_code:
    if root_type in parent2child_dict.keys():
        level_3_code += parent2child_dict[root_type]

#print(len(level_3_code))

level_4_code = []
for root_type in level_3_code:
    if root_type in parent2child_dict.keys():
        level_4_code += parent2child_dict[root_type]

#print(len(level_4_code))

level_5_code = []
for root_type in level_4_code:
    if root_type in parent2child_dict.keys():
        level_5_code += parent2child_dict[root_type]

#print(len(level_5_code))

level_6_code = []
for root_type in level_5_code:
    if root_type in parent2child_dict.keys():
        level_6_code += parent2child_dict[root_type]

#print(len(level_6_code))

level_7_code = []
for root_type in level_6_code:
    if root_type in parent2child_dict.keys():
        level_7_code += parent2child_dict[root_type]

#print(len(level_7_code))

level_8_code = []
for root_type in level_7_code:
    if root_type in parent2child_dict.keys():
        level_8_code += parent2child_dict[root_type]

#print(len(level_8_code))

level_9_code = []
for root_type in level_8_code:
    if root_type in parent2child_dict.keys():
        level_9_code += parent2child_dict[root_type]

#print(len(level_9_code))

level_10_code = []
for root_type in level_9_code:
    if root_type in parent2child_dict.keys():
        level_10_code += parent2child_dict[root_type]

#print(len(level_10_code))

level_11_code = []
for root_type in level_10_code:
    if root_type in parent2child_dict.keys():
        level_11_code += parent2child_dict[root_type]

#print(len(level_11_code))

level_12_code = []
for root_type in level_11_code:
    if root_type in parent2child_dict.keys():
        level_12_code += parent2child_dict[root_type]

#print(len(level_12_code))

level_13_code = []
for root_type in level_12_code:
    if root_type in parent2child_dict.keys():
        level_13_code += parent2child_dict[root_type]

#print(len(level_13_code))

level_14_code = []
for root_type in level_13_code:
    if root_type in parent2child_dict.keys():
        level_14_code += parent2child_dict[root_type]

#print(len(level_14_code))

def filter_level_code(level_9_code, level):
    filtered_level_set = []
    for cur_code in level_9_code:
        flag = False
        for round in range(9-level):
            if cur_code in child2parent_dict.keys():
                cur_code = child2parent_dict[cur_code][0]
            else:
                flag = True
                break
        if not flag:
            filtered_level_set += [cur_code]
    return list(set(filtered_level_set))
            


level_10_lower_code = level_10_code + level_11_code + level_12_code + level_13_code + level_14_code
#level_nodes = [root_set, level_1_code, level_2_code, level_3_code, level_4_code, level_5_code, level_6_code, level_7_code, level_8_code, level_9_lower_code]
#level_nodes = [filter_level_code(level_9_code, i) for i in range(6, 9)] + [level_9_code] + [level_10_lower_code]
level_nodes = [level_6_code, level_7_code, level_8_code, level_9_code, level_10_lower_code]

for level_node in level_nodes:
    print(len(level_node))
#print(len(child2parent_dict.keys()))

for i in range(len(level_nodes)):
    cur_nodes = level_nodes[i]
    for node in cur_nodes:
        print(id2name_dict[node], i)


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
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 0:
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            total_size = len(sampled_nodes)
            for node in tqdm(sampled_nodes):
                
                cur_template = ('medical-oae', id2name_dict[child2parent_dict[node][0]], id2name_dict[node], cur_level, 'level-positive')    
                
                csv_writer.writerow(cur_template)
            print('size of the positive set', total_size, cur_level)
        elif question_mode == 1: # negative hard question
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            total_size = len(sampled_nodes)
            for node in tqdm(sampled_nodes):
                cur_p = id2name_dict[random.choice(nodes_uncles_dict[node])]
                
                cur_template = ('medical-oae', cur_p, id2name_dict[node], cur_level, 'level-negative-hard')    
                
                csv_writer.writerow(cur_template)
            print('size of the negative set', total_size, cur_level)

        elif question_mode == 2:
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            total_size = len(sampled_nodes)
            for node in tqdm(sampled_nodes):
                cur_p = id2name_dict[random.choice(list(set(level_nodes[cur_level-1])-set(child2parent_dict[node])))]
                
                cur_template = ('medical-oae', cur_p, id2name_dict[node], cur_level, 'level-negative-easy') 
                
                csv_writer.writerow(cur_template)
            print('size of the negative set', total_size, cur_level)




num_question_samples = [100000, 100000, 100000, 100000]

file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv']


for cur_question_mode in range(3):
    out_path = './TaxoGlimpse/question_pools/medical-oae/level/level_question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    
    for cur_level in range(1, len(num_question_samples)+1):
        nodes_uncles_dict = get_cur_level_uncles(level_nodes[cur_level], level_nodes, cur_level) # level 1是从level 1 到root
        sample_question_pool(level_nodes[cur_level], level_nodes, nodes_uncles_dict, sample_size=num_question_samples[cur_level-1], cur_level=cur_level, out_path=out_path, question_mode=cur_question_mode)

