import csv
import random
from tqdm import tqdm

def process_link_to_name(link):
    return link.split('/')[-1]

def clean_name(name):
    if "#" in name:
        return False
    else:
        return True

child2parent_dict = {}
parent2child_dict = {}
root_set = []

# 打开CSV文件
with open('./schemaorg-current-https-types-v26.csv', 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)

    # 遍历文件中的每一行
    for idx, row in enumerate(csv_reader):
        # 打印每一行的数据
        if idx > 0:
            child = process_link_to_name(row[0])
            parent = process_link_to_name(row[3])
            if not parent:
                root_set.append(child)
                continue
            if not child:
                continue
            if not clean_name(parent):
                continue
            if not clean_name(child):
                continue
            
            if not child in child2parent_dict.keys():
                child2parent_dict[child] = [parent]
            else:
                child2parent_dict[child].append(parent)
            if not parent in parent2child_dict.keys():
                parent2child_dict[parent] = [child]
            else:
                parent2child_dict[parent].append(child)

for cur_parent in parent2child_dict.keys():
    if cur_parent in child2parent_dict.keys():
        continue
    else:
        if not cur_parent in root_set:
            root_set.append(cur_parent)

filtered_root_set = []
for root in root_set:
    if root in parent2child_dict.keys():
        filtered_root_set.append(root)

print(len(filtered_root_set))

level_1 = []
for item in filtered_root_set:
    if item in parent2child_dict.keys():
        level_1+=parent2child_dict[item]

print(len(level_1))

level_2 = []
for item in level_1:
    if item in parent2child_dict.keys():
        level_2 += parent2child_dict[item]

print(len(level_2))

level_3 = []
for item in level_2:
    if item in parent2child_dict.keys():
        level_3 += parent2child_dict[item]

print(len(level_3))

level_4 = []
for item in level_3:
    if item in parent2child_dict.keys():
        level_4 += parent2child_dict[item]

print(len(level_4))

level_5 = []
for item in level_4:
    if item in parent2child_dict.keys():
        level_5 += parent2child_dict[item]

print(len(level_5))

level_6 = []
for item in level_5:
    if item in parent2child_dict.keys():
        level_6 += parent2child_dict[item]

print(len(level_6))


level_nodes = [filtered_root_set, level_1, level_2, level_3, level_4, level_5, level_6]

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
        if question_mode == 0: # hard mcq questions
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            
            total_size = len(sampled_nodes)
            for node in tqdm(sampled_nodes):
                if len(nodes_uncles_dict[node]) < 3: # 若entity数量不足 用同级的其他entity进行补足
                    selected_mcqs = nodes_uncles_dict[node] + random.sample(list(set(level_nodes[cur_level])-set([node])), int(3-len(nodes_uncles_dict[node])))
                else:
                    selected_mcqs = random.sample(nodes_uncles_dict[node], 3)
                cur_template = ('general-schema', child2parent_dict[node][0], node, cur_level, 'mcq-negative-hard', selected_mcqs[0], selected_mcqs[1], selected_mcqs[2])
                csv_writer.writerow(cur_template)
            print('size of the mcq questions', total_size, cur_level)

        elif question_mode == 1:
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            
            total_size = len(sampled_nodes)
            for node in tqdm(sampled_nodes):
                if len(nodes_uncles_dict[node]) < 3:
                    selected_mcqs = nodes_uncles_dict[node] + random.sample(list(set(level_nodes[cur_level])-set([node])), int(3-len(nodes_uncles_dict[node])))
                else:
                    selected_mcqs = random.sample(list(set(level_nodes[cur_level-1])-set(child2parent_dict[node])), 3)
                cur_template = ('general-schema', child2parent_dict[node][0], node, cur_level, 'mcq-negative-easy', selected_mcqs[0], selected_mcqs[1], selected_mcqs[2])
                csv_writer.writerow(cur_template)
            print('size of the mcq questions', total_size, cur_level)

num_question_samples = [100000, 100000, 100000, 100000, 100000]

file_name = ['mcq_hard.csv', 'mcq_easy.csv']


for cur_question_mode in range(2):
    out_path = 'TaxoGlimpse/question_pools/general-schema/mcq/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type', 'mcq-parent1', 'mcq-parent2', 'mcq-parent3'])  # 写入表头
    
    for cur_level in range(1, len(num_question_samples)+1):
        nodes_uncles_dict = get_cur_level_uncles(level_nodes[cur_level], level_nodes, cur_level) # level 1是从level 1 到root
        sample_question_pool(level_nodes[cur_level], level_nodes, nodes_uncles_dict, sample_size=num_question_samples[cur_level-1], cur_level=cur_level, out_path=out_path, question_mode=cur_question_mode)
