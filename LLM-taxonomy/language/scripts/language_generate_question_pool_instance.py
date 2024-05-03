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

def get_root_name(node, language_dict):
    while node.parent:
        node = node.parent
        cur_name = node.clade.name
    return language_dict[cur_name]

def setup_seed(seed):
    random.seed(seed)

def get_leaf_nodes(node):
    if not node.children:
        return [node]  # 如果节点没有子节点，则为叶子节点

    leaf_nodes = []
    for child in node.children:
        leaf_nodes.extend(get_leaf_nodes(child))  # 递归获取子节点的叶子节点
    return leaf_nodes

def find_ancestor_path(node, target_leaf, path):
    if node.clade.name == target_leaf.clade.name:
        return path + [node]

    if not node.children:
        return []

    for child in node.children:
        ancestor_path = find_ancestor_path(child, target_leaf, path + [node])
        if ancestor_path:
            return ancestor_path

    return []

def get_tree_path():
    tree_path = '/export/data/LLM-benchmark-project-KB/LLM-taxonomy/language/glottolog-glottolog-cldf-59a612c/cldf/classification.nex'
    tree_paths = []
    custom_tree_list = parse_tree(tree_path)
    for i, custom_tree in enumerate(custom_tree_list):
        leaf_nodes = get_leaf_nodes(custom_tree)
        for leaf_node in leaf_nodes:
            tree_paths.append(find_ancestor_path(custom_tree, leaf_node, []))
    return tree_paths

def sample_question_pool_instance(tree_paths, language_dict, out_path, max_levels, question_mode = 0):
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 1 or question_mode == 2: #negative
            for cur_level in range(max_levels):
                root_set = []
                for tree_path in tree_paths:
                    tree_length = len(tree_path)
                    if tree_length <= cur_level + 1:
                        continue
                    else:
                        cur_root = tree_path[cur_level]
                        if cur_root in root_set:
                            continue
                        else:
                            root_set.append(get_name(cur_root, language_dict))

                sampled_pairs = []
                for tree_path in tree_paths:
                    tree_length = len(tree_path)
                    if tree_length <= cur_level + 1:
                        continue
                    else:
                        sampled_pairs.append((tree_path[cur_level], tree_path[-1]))
            
                num_pairs = 0
                for root, sub in tqdm(sampled_pairs):
                    cur_subtype = get_name(sub, language_dict)
                    cur_root_base = root
                    if (not cur_level == 0) and question_mode == 1: # hard start
                        cur_grand = get_parent_node(cur_root_base) 
                        hard_root_set = cur_grand.children
                        if not len(hard_root_set) < 2:
                            root_set = []
                            for hard_root in hard_root_set:
                                root_set.append(get_name(hard_root, language_dict)) 
                        else:
                            continue
                    
                    cur_root_set = list(set(root_set)-set([get_name(cur_root_base, language_dict)]))
                    
                    cur_root = random.choice(cur_root_set)
                    num_pairs += 1
                    if question_mode == 1:
                        cur_template = ('language-glottolog', cur_root, cur_subtype, cur_level, 'level-negative-hard')
                    else:
                        cur_template = ('language-glottolog', cur_root, cur_subtype, cur_level, 'level-negative-easy')
                    csv_writer.writerow(cur_template)                
                print('size of the negative set', num_pairs, cur_level)
        elif question_mode ==0:
            for cur_level in range(max_levels):

                sampled_pairs = []
                for tree_path in tree_paths:
                    tree_length = len(tree_path)
                    if tree_length <= cur_level + 1:
                        continue
                    else:
                        sampled_pairs.append((tree_path[cur_level], tree_path[-1]))
            
                for root, sub in tqdm(sampled_pairs):
                    cur_subtype = get_name(sub, language_dict)
                    cur_root = get_name(root, language_dict)
                    cur_template = ('language-glottolog', cur_root, cur_subtype, cur_level, 'level-positive')
                    csv_writer.writerow(cur_template)
                print('size of the positive set', len(sampled_pairs), cur_level)
        


seed = 20
print('current seed', seed)
setup_seed(seed)

language_dict = get_language_dict(language_path)

#deepest_depth = find_deepest_tree_depth(custom_tree_list)

#nodes_at_level_one, nodes_at_level_two, nodes_at_level_three, nodes_at_level_four, nodes_at_deeper = get_nodes_by_levels(custom_tree_list, deepest_depth)
#nodes_at_diff_levels = [nodes_at_level_one, nodes_at_level_two, nodes_at_level_three, nodes_at_level_four, nodes_at_deeper]
file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv', 'positive_to_root.csv', 'negative_to_root.csv']
#num_question_samples = [100000, 100000, 100000, 100000, 100000]

tree_paths = get_tree_path()
for cur_question_mode in range(3):
    out_path = 'TaxoGlimpse/question_pools/language-glottolog/instance_full/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    
    sample_question_pool_instance(tree_paths, language_dict, out_path, 5, question_mode = cur_question_mode)
