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

print(len(root_set))

level_1_code = []
for root_type in root_set:
    if root_type in parent2child_dict.keys():
        level_1_code += parent2child_dict[root_type]

print(len(level_1_code))

level_2_code = []
for root_type in level_1_code:
    if root_type in parent2child_dict.keys():
        level_2_code += parent2child_dict[root_type]

print(len(level_2_code))

level_3_code = []
for root_type in level_2_code:
    if root_type in parent2child_dict.keys():
        level_3_code += parent2child_dict[root_type]

print(len(level_3_code))

level_4_code = []
for root_type in level_3_code:
    if root_type in parent2child_dict.keys():
        level_4_code += parent2child_dict[root_type]

print(len(level_4_code))

level_5_code = []
for root_type in level_4_code:
    if root_type in parent2child_dict.keys():
        level_5_code += parent2child_dict[root_type]

print(len(level_5_code))

level_6_code = []
for root_type in level_5_code:
    if root_type in parent2child_dict.keys():
        level_6_code += parent2child_dict[root_type]

print(len(level_6_code))

level_7_code = []
for root_type in level_6_code:
    if root_type in parent2child_dict.keys():
        level_7_code += parent2child_dict[root_type]

print(len(level_7_code))

level_8_code = []
for root_type in level_7_code:
    if root_type in parent2child_dict.keys():
        level_8_code += parent2child_dict[root_type]

print(len(level_8_code))

level_9_code = []
for root_type in level_8_code:
    if root_type in parent2child_dict.keys():
        level_9_code += parent2child_dict[root_type]

print(len(level_9_code))

level_10_code = []
for root_type in level_9_code:
    if root_type in parent2child_dict.keys():
        level_10_code += parent2child_dict[root_type]

print(len(level_10_code))

level_11_code = []
for root_type in level_10_code:
    if root_type in parent2child_dict.keys():
        level_11_code += parent2child_dict[root_type]

print(len(level_11_code))

level_12_code = []
for root_type in level_11_code:
    if root_type in parent2child_dict.keys():
        level_12_code += parent2child_dict[root_type]

print(len(level_12_code))

level_13_code = []
for root_type in level_12_code:
    if root_type in parent2child_dict.keys():
        level_13_code += parent2child_dict[root_type]

print(len(level_13_code))

level_14_code = []
for root_type in level_13_code:
    if root_type in parent2child_dict.keys():
        level_14_code += parent2child_dict[root_type]

print(len(level_14_code))


level_10_lower_code = level_10_code + level_11_code + level_12_code + level_13_code + level_14_code
level_nodes = [level_6_code, level_7_code, level_8_code, level_9_code, level_10_lower_code]

for level_node in level_nodes:
    print(len(level_node))
#print(len(child2parent_dict.keys()))

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
        
        if question_mode == 0: # MCQ hard
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            total_size = len(sampled_nodes)
            for node in tqdm(sampled_nodes):
                flag = False
                if 'http' in id2name_dict[child2parent_dict[node][0]] or 'http' in id2name_dict[node]:
                    total_size -= 1
                    continue
                if len(nodes_uncles_dict[node]) < 3:
                    cur_p = nodes_uncles_dict[node] + random.sample(list(set(level_nodes[cur_level])-set([node])), int(3-len(nodes_uncles_dict[node])))
                    retry = 0

                    while 'http' in cur_p[0] or 'http' in cur_p[1] or 'http' in cur_p[2]:
                        cur_p = nodes_uncles_dict[node] + random.sample(list(set(level_nodes[cur_level])-set([node])), int(3-len(nodes_uncles_dict[node])))
                        retry += 1
                        if retry == 10:
                            flag = True
                            break
                    if flag:
                        continue
                else:
                    cur_p = [id2name_dict[item] for item in random.sample(nodes_uncles_dict[node], 3)]
                    retry = 0
                    while 'http' in cur_p[0] or 'http' in cur_p[1] or 'http' in cur_p[2]:
                    
                        cur_p = [id2name_dict[item] for item in random.sample(nodes_uncles_dict[node], 3)]
                        retry += 1
                        if retry == 10:
                            flag = True
                            break
                    if flag:
                        continue

                cur_template = ('medcial-oae', id2name_dict[child2parent_dict[node][0]], id2name_dict[node], cur_level, 'mcq-negative-hard', cur_p[0], cur_p[1], cur_p[2])    
                
                csv_writer.writerow(cur_template)
            print('size of the MCQ set', total_size, cur_level)

        elif question_mode == 1: # MCQ easy
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            total_size = len(sampled_nodes)
            for node in tqdm(sampled_nodes):
                flag = False
                if 'http' in id2name_dict[child2parent_dict[node][0]] or 'http' in id2name_dict[node]:
                    total_size -= 1
                    continue
                if len(nodes_uncles_dict[node]) < 3:
                    cur_p = nodes_uncles_dict[node] + random.sample(list(set(level_nodes[cur_level])-set([node])), int(3-len(nodes_uncles_dict[node])))
                    retry = 0
                    while 'http' in cur_p[0] or 'http' in cur_p[1] or 'http' in cur_p[2]:
                    
                        cur_p = nodes_uncles_dict[node] + random.sample(list(set(level_nodes[cur_level])-set([node])), int(3-len(nodes_uncles_dict[node])))
                        retry += 1
                        if retry == 10:
                            flag = True
                            break
                    if flag:
                        continue
                else:
                    cur_p = [id2name_dict[item] for item in random.sample(list(set(level_nodes[cur_level-1])-set(child2parent_dict[node])), 3)]
                    retry = 0
                    while 'http' in cur_p[0] or 'http' in cur_p[1] or 'http' in cur_p[2]:
                    
                        cur_p = [id2name_dict[item] for item in random.sample(list(set(level_nodes[cur_level-1])-set(child2parent_dict[node])), 3)]
                        retry += 1
                        if retry == 10:
                            flag = True
                            break
                    if flag:
                        continue
                cur_template = ('medical-oae', id2name_dict[child2parent_dict[node][0]], id2name_dict[node], cur_level, 'mcq-negative-easy', cur_p[0], cur_p[1], cur_p[2])    
                
                csv_writer.writerow(cur_template)
            print('size of the MCQ set', total_size, cur_level)


num_question_samples = [100000, 100000, 100000, 100000]

file_name = ['mcq_hard.csv', 'mcq_easy.csv']


for cur_question_mode in range(2):
    out_path = 'TaxoGlimpse/question_pools/medical-oae/mcq/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type', 'mcq-parent1', 'mcq-parent2', 'mcq-parent3'])  # 写入表头
    
    for cur_level in range(1, len(num_question_samples)+1):
        nodes_uncles_dict = get_cur_level_uncles(level_nodes[cur_level], level_nodes, cur_level) # level 1是从level 1 到root
        sample_question_pool(level_nodes[cur_level], level_nodes, nodes_uncles_dict, sample_size=num_question_samples[cur_level-1], cur_level=cur_level, out_path=out_path, question_mode=cur_question_mode)

