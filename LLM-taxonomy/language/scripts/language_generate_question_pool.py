import os
import csv
from io import StringIO
from Bio import Phylo
import openai
import random
from tqdm import tqdm
import time


family_path = 'TaxoGlimpse/LLM-taxonomy/language/glottolog-glottolog-cldf-59a612c/cldf/trees.csv'
tree_path = 'TaxoGlimpse/LLM-taxonomy/language/glottolog-glottolog-cldf-59a612c/cldf/classification.nex'
language_path = 'TaxoGlimpse/LLM-taxonomy/language/glottolog-glottolog-cldf-59a612c/cldf/languages.csv'

def get_family_languages(family_path):
    family_languages = []
    family_codes = []
    family_dict = {}
    with open(family_path, 'r') as f:
        for idx, row in enumerate(csv.reader(f, skipinitialspace=True)):
            if idx == 0:
                continue
            cur_family = row[2]
            family_language = cur_family.split('family ')[-1]
            family_languages.append(family_language)
            family_codes.append(row[0])
            family_dict[row[0]] = family_language
    print(family_languages)
    print(family_codes)
    return family_dict

def get_language_dict(language_path):
    language_dict = {}
    with open(language_path,'r') as f:
        for idx, row in enumerate(csv.reader(f, skipinitialspace=True)):
            if idx == 0:
                continue
            cur_language = row[1]
            cur_code = row[0]
            language_dict[cur_code] = cur_language
    return language_dict

class MyTreeNode:
    def __init__(self, clade, parent=None):
        self.clade = clade
        self.parent = parent
        self.children = []

# define custom tree
def build_custom_tree(node, parent=None):
    custom_node = MyTreeNode(node, parent)
    if hasattr(node, 'clades'):
        for child in node.clades:
            custom_node.children.append(build_custom_tree(child, custom_node))
    return custom_node

# get all nodes at level
def get_nodes_at_level(custom_tree, level):
    nodes_at_level = []

    def dfs(node, current_level):
        if current_level == level:
            nodes_at_level.append(node)
            return
        for child in node.children:
            dfs(child, current_level + 1)

    dfs(custom_tree, 0)
    return nodes_at_level

def find_depth(node):
    if not node:
        return 0
    elif not node.children:
        return 1
    else:
        return 1 + max(find_depth(child) for child in node.children)

def find_deepest_tree_depth(custom_trees):
    max_depth = 0
    depth_dict = {}
    for tree in custom_trees:
        depth = find_depth(tree)
        if not depth in depth_dict.keys():
            depth_dict[depth] = 1
        else:
            depth_dict[depth] += 1
        max_depth = max(max_depth, depth)
    sorted_keys = sorted(depth_dict.keys())

    # 按排序后的键访问字典值
    for key in sorted_keys:
        value = depth_dict[key]
        #print(f"Key: {key}, Value: {value}")
    return max_depth

def get_name(tree_node, language_dict):
    return language_dict[tree_node.clade.name]

def get_parent(tree_node, language_dict):
    return language_dict[tree_node.parent.clade.name]

def get_parent_node(tree_node):
    return tree_node.parent

def get_nodes_by_levels(custom_tree_list, deepest_depth):
    nodes_at_level_one = []
    nodes_at_level_two = []
    nodes_at_level_three = []
    nodes_at_level_four = []
    nodes_at_deeper = []

    for custom_tree in custom_tree_list:
        for depth in range(1, deepest_depth+1):
            if depth == 1:
                nodes_at_level_one += get_nodes_at_level(custom_tree, depth)
            elif depth == 2:
                nodes_at_level_two += get_nodes_at_level(custom_tree, depth)
            elif depth == 3:
                nodes_at_level_three += get_nodes_at_level(custom_tree, depth)
            elif depth == 4: #可以调整成之后所有 先看看前面几层的表现
                nodes_at_level_four += get_nodes_at_level(custom_tree, depth)
            elif depth > 4: #可以调整成之后所有 先看看前面几层的表现
                nodes_at_deeper += get_nodes_at_level(custom_tree, depth)
    
    return nodes_at_level_one, nodes_at_level_two, nodes_at_level_three, nodes_at_level_four, nodes_at_deeper

def parse_tree(tree_path):
    tree_list = []
    with open(tree_path, 'r') as f:
        Lines = f.readlines()
    for line in Lines:
        line = line.strip()
        if line[:4] == 'tree':
            nexus_tree = line.split('[&R] ')[-1]
            tree_data = StringIO(nexus_tree)
            tree = Phylo.read(tree_data, "newick")
            custom_tree = build_custom_tree(tree.clade)
            tree_list.append(custom_tree)
    return tree_list


def filter_list(nodes_list, language_dict):
    filtered_list = []
    sudo_list = ['Sign Language', 'Bookkeeping', 'Speech Register', 'Unattested', 'Unclassifiable', 'Artificial', 'Unclassified', 'CSLic']
    for node in nodes_list:
        flag = False
        for sudo_node in sudo_list:
            if (sudo_node in get_parent(node, language_dict)) or (sudo_node in get_name(node, language_dict)):
                flag = True
        if flag:
            continue
        else:
            filtered_list.append(node)
    return filtered_list


def sample_question_pool(nodes_level_list, language_dict, cur_level, sample_size, out_path, question_mode = 0):
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 1 or question_mode == 2: #negative
            root_set = []
            for node in nodes_level_list:
                cur_root = get_parent(node, language_dict)
                if cur_root in root_set:
                    continue
                else:
                    root_set.append(cur_root)
            if len(nodes_level_list) < sample_size:
                sampled_nodes = nodes_level_list
            else:
                sampled_nodes = random.sample(nodes_level_list, sample_size)
            num_pairs = 0
            for node in tqdm(sampled_nodes):
                cur_subtype = get_name(node, language_dict)
                cur_root_base = get_parent(node, language_dict)

                if (not cur_level == 1) and question_mode == 1: # hard start
                    cur_root_base = get_parent_node(node)
                    cur_grand = get_parent_node(cur_root_base) 
                    hard_root_set = cur_grand.children
                    if not len(hard_root_set) < 2:
                        root_set = []
                        for hard_root in hard_root_set:
                            root_set.append(get_name(hard_root, language_dict)) 
                    else:
                        continue
                if question_mode == 1:
                    if cur_level == 1:
                        cur_root_set = list(set(root_set)-set([cur_root_base]))
                    else:
                        cur_root_set = list(set(root_set)-set([get_name(cur_root_base, language_dict)]))
                else:
                    cur_root_set = list(set(root_set)-set([cur_root_base]))
                cur_root = random.choice(cur_root_set)
                num_pairs += 1
                if question_mode == 1:
                    cur_template = ('language-glottolog', cur_root, cur_subtype, cur_level, 'level-negative-hard')
                else:
                    cur_template = ('language-glottolog', cur_root, cur_subtype, cur_level, 'level-negative-easy')
                csv_writer.writerow(cur_template)                
            print('size of the negative set', num_pairs, cur_level)
        elif question_mode ==0:
            if len(nodes_level_list) < sample_size:
                sampled_nodes = nodes_level_list
            else:
                sampled_nodes = random.sample(nodes_level_list, sample_size)
            for node in tqdm(sampled_nodes):
                cur_subtype = get_name(node, language_dict)
                cur_root = get_parent(node, language_dict)
                cur_template = ('language-glottolog', cur_root, cur_subtype, cur_level, 'level-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_nodes), cur_level)
        elif question_mode == 3:
            if len(nodes_level_list) < sample_size:
                sampled_nodes = nodes_level_list
            else:
                sampled_nodes = random.sample(nodes_level_list, sample_size)
            for node in tqdm(sampled_nodes):
                cur_subtype = get_name(node, language_dict)
                cur_root = get_root_name(node, language_dict)
                cur_template = ('language-glottolog', cur_root, cur_subtype, cur_level, 'toroot-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_nodes), cur_level)
        elif question_mode == 4:
            root_set = []
            for node in nodes_level_list:
                cur_root = get_root_name(node, language_dict)
                if cur_root in root_set:
                    continue
                else:
                    root_set.append(cur_root)
            if len(nodes_level_list) < sample_size:
                sampled_nodes = nodes_level_list
            else:
                sampled_nodes = random.sample(nodes_level_list, sample_size)
            num_pairs = 0
            for node in tqdm(sampled_nodes):
                cur_subtype = get_name(node, language_dict)
                cur_root_base = get_root_name(node, language_dict)
                cur_root_set = list(set(root_set)-set([cur_root_base]))
                cur_root = random.choice(cur_root_set)
                num_pairs += 1
                cur_template = ('language-glottolog', cur_root, cur_subtype, cur_level, 'toroot-negative')
                csv_writer.writerow(cur_template)                
            print('size of the negative set', num_pairs, cur_level)

def get_root_name(node, language_dict):
    while node.parent:
        node = node.parent
        cur_name = node.clade.name
    return language_dict[cur_name]

def setup_seed(seed):
    random.seed(seed)

seed = 20
print('current seed', seed)
setup_seed(seed)

language_dict = get_language_dict(language_path)
custom_tree_list = parse_tree(tree_path)

deepest_depth = find_deepest_tree_depth(custom_tree_list)

nodes_at_level_one, nodes_at_level_two, nodes_at_level_three, nodes_at_level_four, nodes_at_deeper = get_nodes_by_levels(custom_tree_list, deepest_depth)
nodes_at_diff_levels = [nodes_at_level_one, nodes_at_level_two, nodes_at_level_three, nodes_at_level_four, nodes_at_deeper]
file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv', 'positive_to_root.csv', 'negative_to_root.csv']
num_question_samples = [100000, 100000, 100000, 100000, 100000]


for cur_question_mode in range(3):
    out_path = 'TaxoGlimpse/question_pools/language-glottolog/level/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    for cur_node_level in range(len(nodes_at_diff_levels)):
        cur_level = cur_node_level+1
        nodes_at_cur_level = filter_list(nodes_at_diff_levels[cur_node_level], language_dict)
        print('total number of nodes', len(nodes_at_cur_level))
        sample_question_pool(nodes_at_cur_level, language_dict, cur_level, num_question_samples[cur_node_level], out_path, question_mode = cur_question_mode)


