import os
import csv
import random
from tqdm import tqdm
import time
import pickle
from collections import deque

class TreeNode:
    def __init__(self, headers, amazon_base_domain='https://www.browsenodes.com', page='https://www.browsenodes.com/amazon.com', name='root', is_leaf=False):
        self.name = name
        self.children = []
        self.is_leaf = is_leaf
        self.page = page
        self.base_domain = amazon_base_domain
        self.headers = headers

        response = requests.get(self.page, headers=headers)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        raw_links = soup.find_all('tr')

        if 'leaf' in str(raw_links):
            self.is_leaf = True
        else:
            self.add_children(raw_links)
        
        print('cur node', self.name)
        print('cur children')
        for child in self.children:
            print(child.name)
        print('===================')

    def add_children(self, raw_links):
        for item in raw_links:
            identified_node = item.find_all('td')
            if len(identified_node) == 0:
                continue
            splitted = str(identified_node).split('>')
            name = splitted[1]
            for splitted_item in splitted:
                if 'read-more' in str(splitted_item):
                    link = str(splitted_item)
            name = name.split('<')[0].strip()
            name = self.clean_name(name)
            link = self.base_domain+link.split('href=')[1][1:-1]
            cur_node = TreeNode(page=link, name=name, headers=self.headers)
            self.children.append(cur_node)

    def clean_name(self, string):
        return string.replace('&amp;', 'and')


def get_all_trees():
    node_names = ['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 'Books', 'Collectibles and Fine Arts', 'Electronics', 'Clothing, Shoes and Jewelry', 'Clothing, Shoes and Jewelry - Baby', 'Clothing, Shoes and Jewelry - Boys', 'Clothing, Shoes and Jewelry - Girls', 'Clothing, Shoes and Jewelry - Men', 'Clothing, Shoes and Jewelry - Women', 'Gift Cards', 'Grocery and Gourmet Food', 'Handmade', 'Health and Personal Care', 'Home and Kitchen', 'Industrial and Scientific', 'Kindle Store', 'Patio, Lawn and Garden', 'Luggage and Travel Gear', 'Magazine Subscriptions', 'Apps and Games', 'Movies and TV', 'Digital Music', 'CDs and Vinyl', 'Musical Instruments', 'Office Products', 'Computers', 'Pet Supplies', 'Software', 'Sports and Outdoors', 'Tools and Home Improvement', 'Toys and Games', 'Amazon Instant Video', 'Vehicles', 'Video Games', 'Wine', 'Cell Phones and Accessories']
    all_trees = []
    for idx in range(len(node_names)):
        with open('TaxoGlimpse/LLM-taxonomy/shopping/data/browsenodes/'+str(idx)+'.pkl', 'rb') as pickle_file:
            cur_tree = pickle.load(pickle_file)
        all_trees.append(cur_tree)
    return all_trees

def get_nodes_and_siblings_at_level(root, level):
    exclusion = ['Books', 'Collectibles and Fine Arts', 'Handmade', 'Magazine Subscriptions', 'Apps and Games', 'Movies and TV', 'CDs and Vinyl', 'Digital Music', 'Amazon Instant Video', 'Clothing, Shoes and Jewelry']
    partial_exclusion = ['Baby', 'Clothing, Shoes and Jewelry - Baby', 'Clothing, Shoes and Jewelry - Boys', 'Clothing, Shoes and Jewelry - Girls', 'Clothing, Shoes and Jewelry - Men', 'Clothing, Shoes and Jewelry - Women']

    if not root:
        return []

    queue = deque([(root, None, 0)])
    nodes_at_level = []

    while queue:
        node, parent, current_level = queue.popleft()

        if current_level == level:
            if (root.name in exclusion) or (root.name in partial_exclusion and level==1):
                siblings = [child.name + ' (' + root.name + ')'  for child in parent.children if child != node] if parent else []
            else:
                siblings = [child.name for child in parent.children if child != node] if parent else []
            nodes_at_level.append((node, siblings))
        elif current_level > level:
            break

        for child in node.children:
            queue.append((child, node, current_level + 1))

    return nodes_at_level


def get_nodes_at_level(root, level):
    if not root:
        return []

    queue = deque([(root, 0)])
    #nodes_at_level = []
    node_objects_at_level = []

    while queue:
        node, current_level = queue.popleft()

        if current_level == level:
            #nodes_at_level.append(node.name)
            node_objects_at_level.append(node)
        elif current_level > level:
            break

        for child in node.children:
            queue.append((child, current_level + 1))

    return node_objects_at_level

def process_trees_MCQ(all_trees, level): # level这里从0开始数
    exclusion = ['Books', 'Collectibles and Fine Arts', 'Handmade', 'Magazine Subscriptions', 'Apps and Games', 'Movies and TV', 'CDs and Vinyl', 'Digital Music', 'Amazon Instant Video', 'Clothing, Shoes and Jewelry', 'Kindle Store']
    partial_exclusion = ['Baby', 'Clothing, Shoes and Jewelry - Baby', 'Clothing, Shoes and Jewelry - Boys', 'Clothing, Shoes and Jewelry - Girls', 'Clothing, Shoes and Jewelry - Men', 'Clothing, Shoes and Jewelry - Women']

    cur_level_nodes = []
    cur_level_nodes_str = []

    cur_level_nodes_uncles = []

    all_trees_names = [cur_tree.name for cur_tree in all_trees]

    if level < 3:
        for tree_idx, cur_tree in enumerate(all_trees):
            cur_level_root_nodes = get_nodes_and_siblings_at_level(cur_tree, level)
            for cur_root_node in cur_level_root_nodes:
                cur_sub_nodes = cur_root_node[0].children
                for cur_sub_node in cur_sub_nodes:
                    if (cur_tree.name in exclusion):
                        if level > 0:
                            cur_pair = (cur_root_node[0].name + ' (' + cur_tree.name + ')' , cur_sub_node.name + ' (' + cur_tree.name + ')' )
                        else:
                            cur_pair = (cur_root_node[0].name, cur_sub_node.name + ' (' + cur_tree.name + ')' )
                    elif ((cur_tree.name in partial_exclusion) and level == 0):
                        cur_pair = (cur_root_node[0].name, cur_sub_node.name + ' (' + cur_tree.name + ')' )
                    else:
                        cur_pair = (cur_root_node[0].name, cur_sub_node.name)
                    if not str(cur_pair) in cur_level_nodes_str:
                        cur_level_nodes_str.append(str(cur_pair))
                        cur_level_nodes.append(cur_pair)
                        if level == 0:
                            cur_all_trees = all_trees_names.copy()
                            cur_all_trees.pop(tree_idx)
                            cur_level_nodes_uncles.append(cur_all_trees)
                        else:
                            cur_level_nodes_uncles.append(cur_root_node[1])
    else:
        i = 0
        for node_idx in range(level, 9):
            for cur_tree in all_trees:
                cur_level_root_nodes = get_nodes_and_siblings_at_level(cur_tree, node_idx)
                for cur_root_node in cur_level_root_nodes:
                    cur_sub_nodes = cur_root_node[0].children
                    for cur_sub_node in cur_sub_nodes:
                        if (cur_tree.name in exclusion):
                            cur_pair = (cur_root_node[0].name + ' (' + cur_tree.name + ')' , cur_sub_node.name + ' (' + cur_tree.name + ')' , node_idx)
                        else:
                            cur_pair = (cur_root_node[0].name, cur_sub_node.name, node_idx)
                        if not str(cur_pair) in cur_level_nodes_str:
                            cur_level_nodes_str.append(str(cur_pair))
                            cur_level_nodes.append(cur_pair)
                            cur_level_nodes_uncles.append(cur_root_node[1])
    print('total number of pairs', len(cur_level_nodes))
    return cur_level_nodes, cur_level_nodes_uncles


def sample_nodes_hard(cur_level_pairs, cur_level_nodes_uncles,sample_size, out_path, cur_level, question_mode = 0):
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 1: # negative question
            if len(cur_level_pairs) < sample_size:
                sampled_idx = range(len(cur_level_pairs))
            else:
                sampled_idx = random.sample(range(len(cur_level_pairs)), sample_size)
            negative_sampled_pairs = []
            for sampled_id in sampled_idx:
                cur_root_set = cur_level_nodes_uncles[sampled_id]
                sampled_pair = cur_level_pairs[sampled_id]
                negative_sampled_pairs.append((random.choice(cur_root_set),sampled_pair[1]))
            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('shopping-amazon', root, child, cur_level+1, 'level-negative-hard')
                csv_writer.writerow(cur_template)
            
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        elif question_mode == 0:
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            for item in tqdm(sampled_pairs):
                root = item[0]
                child = item[1]
                cur_template = ('shopping-amazon', root, child, cur_level+1, 'level-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_pairs), cur_level)
        
        elif question_mode == 2:
            level = cur_level
            if level < 3:
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
            else:
                root_set = []
                for node_idx in range(level, 9):
                    root_set.append([])
                    for pair in cur_level_pairs:
                        if pair[0] in root_set[node_idx-3]:
                            continue
                        else:
                            root_set[node_idx-3].append(pair[0])
                if len(cur_level_pairs) < sample_size:
                    sampled_pairs = cur_level_pairs
                else:
                    sampled_pairs = random.sample(cur_level_pairs, sample_size)
                negative_sampled_pairs = []
                for sampled_pair in sampled_pairs:
                    cur_level_idx = sampled_pair[2]
                    cur_root_set = list(set(root_set[cur_level_idx-3])-set([sampled_pair[0]]))
                    negative_sampled_pairs.append((random.choice(cur_root_set),sampled_pair[1]))
            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('shopping-amazon', root, child, cur_level+1, 'level-negative-easy')
                csv_writer.writerow(cur_template)
            
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        elif question_mode == 3:
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            for item in tqdm(sampled_pairs):
                root = item[0]
                child = item[1]
                cur_template = ('shopping-amazon', root, child, cur_level+1, 'toroot-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_pairs), cur_level)
        elif question_mode == 4:
            level = cur_level
            if level < 3:
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
            else:
                root_set = []
                for node_idx in range(level, 9):
                    root_set.append([])
                    for pair in cur_level_pairs:
                        if pair[0] in root_set[node_idx-3]:
                            continue
                        else:
                            root_set[node_idx-3].append(pair[0])
                if len(cur_level_pairs) < sample_size:
                    sampled_pairs = cur_level_pairs
                else:
                    sampled_pairs = random.sample(cur_level_pairs, sample_size)
                negative_sampled_pairs = []
                for sampled_pair in sampled_pairs:
                    cur_level_idx = sampled_pair[2]
                    cur_root_set = list(set(root_set[cur_level_idx-3])-set([sampled_pair[0]]))
                    negative_sampled_pairs.append((random.choice(cur_root_set),sampled_pair[1]))
            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('shopping-amazon', root, child, cur_level+1, 'toroot-negative')
                csv_writer.writerow(cur_template)
            
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        
def sample_nodes_MCQ(cur_level_pairs, cur_level_nodes_uncles,sample_size, out_path, cur_level, question_mode = 0):
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 0: # MCQ hard
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            for idx, item in enumerate(tqdm(sampled_pairs)):
                root = item[0]
                child = item[1]
                
                if len(cur_level_nodes_uncles[idx]) < 3:
                    sampled_indices = random.sample(list(set(list(range(len(sampled_pairs))))-set([idx])), 3-len(cur_level_nodes_uncles[idx]))
                    sampled_complements = [sampled_pairs[sampled_idx][1] for sampled_idx in sampled_indices]
                    selected_mcqs = cur_level_nodes_uncles[idx] + sampled_complements
                else:
                    selected_mcqs = random.sample(cur_level_nodes_uncles[idx], 3)
                
                cur_template = ('shopping-amazon', root, child, cur_level+1, 'mcq-negative-hard', selected_mcqs[0], selected_mcqs[1], selected_mcqs[2])
                csv_writer.writerow(cur_template)
            print('size of the MCQ set', len(sampled_pairs), cur_level)

        # TODO: update MCQ easy
        
        elif question_mode == 1:
            level = cur_level
            if level < 3:
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
                for idx, item in enumerate(tqdm(sampled_pairs)):
                    root = item[0]
                    child = item[1]
                    cur_root_set = list(set(root_set)-set([item[0]]))
                    selected_mcqs = random.sample(cur_root_set, 3) # 没考虑size小于3 因为不会出现
                    cur_template = ('shopping-amazon', root, child, cur_level+1, 'mcq-negative-easy', selected_mcqs[0], selected_mcqs[1], selected_mcqs[2])
                    csv_writer.writerow(cur_template)
                print('size of the MCQ set', len(sampled_pairs), cur_level)
            else:
                root_set = []
                for node_idx in range(level, 9):
                    root_set.append([])
                    for pair in cur_level_pairs:
                        if pair[0] in root_set[node_idx-3]:
                            continue
                        else:
                            root_set[node_idx-3].append(pair[0])
                if len(cur_level_pairs) < sample_size:
                    sampled_pairs = cur_level_pairs
                else:
                    sampled_pairs = random.sample(cur_level_pairs, sample_size)
                for idx, item in enumerate(tqdm(sampled_pairs)):
                    root = item[0]
                    child = item[1]
                    cur_level_idx = item[2]
                    cur_root_set = list(set(root_set[cur_level_idx-3])-set([item[0]]))
                    selected_mcqs = random.sample(cur_root_set, 3) # 没考虑size小于3 因为不会出现
                    cur_template = ('shopping-amazon', root, child, cur_level+1, 'mcq-negative-easy', selected_mcqs[0], selected_mcqs[1], selected_mcqs[2])
                    csv_writer.writerow(cur_template)
                print('size of the MCQ set', len(sampled_pairs), cur_level)


def setup_seed(seed):
    random.seed(seed)

seed = 20

setup_seed(seed) # 24 for negative and positive questions
print('current seed', seed)

all_trees = get_all_trees()
num_question_samples = [100000, 100000, 100000, 100000]

file_name = ['mcq_hard.csv', 'mcq_easy.csv']

for cur_question_mode in range(2):
    out_path = 'TaxoGlimpse/question_pools/shopping-amazon/mcq/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type', 'mcq-parent1', 'mcq-parent2', 'mcq-parent3'])  # 写入表头
    
    for cur_level in range(len(num_question_samples)):
        cur_level_pairs, cur_level_pairs_uncles = process_trees_MCQ(all_trees, level=cur_level)
        #print(cur_level_pairs[0], cur_level_pairs_uncles[0])
        sample_nodes_MCQ(cur_level_pairs, cur_level_pairs_uncles, sample_size=num_question_samples[cur_level], out_path=out_path, cur_level=cur_level, question_mode=cur_question_mode)
