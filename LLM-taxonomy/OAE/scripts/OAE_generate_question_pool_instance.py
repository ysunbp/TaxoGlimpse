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

def get_paths(root_set, parent2child):
    paths = []  # 存储所有从根节点到叶子节点的路径

    def dfs(node, path):
        path.append(node)  # 将当前节点添加到路径中

        if node not in parent2child:
            # 当前节点是叶子节点，将路径添加到结果中
            paths.append(path[:])  # 创建路径的副本，以避免后续修改影响已添加的路径
        else:
            children = parent2child[node]
            for child in children:
                dfs(child, path)

        path.pop()  # 回溯，移除当前节点

    for root in root_set:
        dfs(root, [])

    return paths


def sample_question_pool_instance(paths, cur_level, out_path, id2name_dict, question_mode, nodes_uncles_dict, sample_size=100000):
    # cur_level from 1 to 10
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)

        if len(paths) < sample_size:
            sampled_paths = paths
        else:
            sampled_paths = random.sample(paths, sample_size)
        
        total_size = len(sampled_paths)
        if question_mode == 0:
            for cur_path in sampled_paths:
                cur_length = len(cur_path)
                if cur_level + 1 > cur_length:
                    continue
                else:
                    cur_child_id = cur_path[-1]
                    cur_parent_id = cur_path[cur_level-1]
                
                    cur_template = ('medical-oae', id2name_dict[cur_parent_id], id2name_dict[cur_child_id], 'instance'+str(cur_level), 'level-positive')
                    csv_writer.writerow(cur_template)
            print('size of the positive set', total_size, cur_level)
        
        elif question_mode == 1:
            for cur_path in sampled_paths:
                cur_length = len(cur_path)
                if cur_level + 1 > cur_length:
                    continue
                else:
                    cur_child_id = cur_path[-1]
                    cur_parent_id = cur_path[cur_level-1]
                    cur_parent_child_id = cur_path[cur_level]
                
                    cur_p = id2name_dict[random.choice(nodes_uncles_dict[cur_parent_child_id])]
                    
                    cur_template = ('medical-oae', cur_p, id2name_dict[cur_child_id], 'instance'+str(cur_level), 'level-negative-hard')
                    csv_writer.writerow(cur_template)
            print('size of the negative set', total_size, cur_level)
        
        elif question_mode == 2:
            for cur_path in sampled_paths:
                cur_length = len(cur_path)
                if cur_level + 1 > cur_length:
                    continue
                else:
                    cur_child_id = cur_path[-1]
                    cur_parent_id = cur_path[cur_level-1]
                    cur_parent_child_id = cur_path[cur_level]
                
                    cur_p = id2name_dict[random.choice(list(set(level_nodes[cur_level-1])-set([cur_parent_id])))]
                    
                    cur_template = ('medical-oae', cur_p, id2name_dict[cur_child_id], 'instance'+str(cur_level), 'level-negative-easy')
                    csv_writer.writerow(cur_template)
            print('size of the negative set', total_size, cur_level)


length = []
for key, value in parent2child_dict.items():
    length += value

root_set = []
for cur_parent in parent2child_dict.keys():
    if cur_parent in child2parent_dict.keys():
        continue
    else:
        if not cur_parent in root_set:
            root_set.append(cur_parent)

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


paths = get_paths(level_nodes[0], parent2child_dict)

file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv']
num_question_samples = [100000, 100000, 100000, 100000]

for cur_question_mode in range(3):
    out_path = 'TaxoGlimpse/question_pools/medical-oae/instance_full/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    
    for cur_level in range(1, len(num_question_samples)+1):
        nodes_uncles_dict = get_cur_level_uncles(level_nodes[cur_level], level_nodes, cur_level)
        sample_question_pool_instance(paths, cur_level, out_path, id2name_dict, cur_question_mode, nodes_uncles_dict)
