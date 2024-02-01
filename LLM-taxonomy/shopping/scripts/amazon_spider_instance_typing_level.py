from bs4 import BeautifulSoup
import requests
from tqdm import trange, tqdm
import os
import random
import time
import pickle
from fake_useragent import UserAgent
from collections import deque
import datetime
import csv


ua = UserAgent()

amazon_base_page = 'https://www.browsenodes.com/amazon.com'
amazon_base_domain = 'https://www.browsenodes.com'


node_names = ['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 'Books', 'Collectibles and Fine Arts', 'Electronics', 'Clothing, Shoes and Jewelry', 'Clothing, Shoes and Jewelry - Baby', 'Clothing, Shoes and Jewelry - Boys', 'Clothing, Shoes and Jewelry - Girls', 'Clothing, Shoes and Jewelry - Men', 'Clothing, Shoes and Jewelry - Women', 'Gift Cards', 'Grocery and Gourmet Food', 'Handmade', 'Health and Personal Care', 'Home and Kitchen', 'Industrial and Scientific', 'Kindle Store', 'Patio, Lawn and Garden', 'Luggage and Travel Gear', 'Magazine Subscriptions', 'Apps and Games', 'Movies and TV', 'Digital Music', 'CDs and Vinyl', 'Musical Instruments', 'Office Products', 'Computers', 'Pet Supplies', 'Software', 'Sports and Outdoors', 'Tools and Home Improvement', 'Toys and Games', 'Amazon Instant Video', 'Vehicles', 'Video Games', 'Wine', 'Cell Phones and Accessories']
node_links = ['https://www.browsenodes.com/amazon.com/browseNodeLookup/2619526011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2617942011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/15690151.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/165797011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/11055981.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/1000.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/4991426011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/493964.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7141124011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147444011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147443011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147442011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147441011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147440011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2864120011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/16310211.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/11260433011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/3760931.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/1063498.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/16310161.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/133141011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/3238155011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/9479199011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/599872.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2350150011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2625374011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/624868011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/301668.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/11965861.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/1084128.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/541966.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2619534011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/409488.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/3375301.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/468240.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/165795011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2858778011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/10677470011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/11846801.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2983386011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2335753011.html']

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


def get_raw_links(page, headers):
    response = requests.get(page, headers=headers)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    content = soup.find_all('tr')
    return content

def clean_name(string):
    return string.replace('&amp;', 'and')

def is_leaf_node(raw_links):
    if 'leaf' in str(raw_links):
        return True
    else:
        return False

def read_product_page(product_link, headers):
    response = requests.get(product_link, headers=headers)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    content = soup.find_all('tr')
    product_names = []
    for item in content:
        identified_node = item.find_all('td')
        if len(identified_node) == 0:
            continue
        splitted = str(identified_node).split('>')
        name = splitted[1]
        name = name.split('<')[0].strip()
        product_names.append(name)
    return product_names

def find_all_leaf_nodes(page, headers):
    base_domain='https://www.browsenodes.com'

    raw_links = get_raw_links(page, headers)
    
    leaf_product_dict={}
    
    def traverse(raw_links, page):
        children_links = []
        if is_leaf_node(raw_links):
            item_id = page.split('/')[-1].split('.')[0]
            product_link = "https://www.commercedna.com/amazon.com/itemSearch/"+str(item_id)+".html"
            leaf_name = str(raw_links).split(' is a leaf node')[0].split('>')[-1]
            
            products = read_product_page(product_link, headers)
            leaf_product_dict[leaf_name] = products
        else:
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
                name = clean_name(name)
                link = base_domain+link.split('href=')[1][1:-1]
                children_links.append((get_raw_links(link, headers), link))
            for child_raw_links, child_page in children_links:
                traverse(child_raw_links, child_page)

    traverse(get_raw_links(page, headers), page)
    return leaf_product_dict

def get_children_names(children_nodes):
    children_names = []
    for child_node in children_nodes:
        children_names.append(child_node.name)
    return children_names


def bfs_traversal(root, cur_leaf_product_keys):
    if not root:
        return []

    queue = deque([root])
    #result = []

    level = 0

    leaf_sibling_mapping = {}

    while queue:
        level_size = len(queue)
        #print(level_size)
        level_result = []

        for _ in range(level_size):
            node = queue.popleft()
            cur_node_children = get_children_names(node.children)
            if len(cur_node_children)>0:
                for cur_leaf_key in cur_leaf_product_keys:
                    if clean_name(cur_leaf_key) in cur_node_children:
                        leaf_sibling_mapping[clean_name(cur_leaf_key)] = list(set(cur_node_children)-set([clean_name(cur_leaf_key)]))

            #level_result.append(node.name)

            for child in node.children:
                queue.append(child)

        #result.append(level_result)
        level += 1
    return leaf_sibling_mapping

# get leaf product files


def setup_seed(seed):
    random.seed(seed)


def find_ancestors(root, target_nodes):
    ancestors = {}
    for node in target_nodes:
        node = clean_name(node)
        ancestors[node] = ()
        _find_ancestors_helper(root, node, [], [], ancestors)

    return ancestors


def _find_ancestors_helper(node, target_node, current_path, current_path_name, ancestors):
    current_path.append(node)
    current_path_name.append(node.name)

    if node.name == target_node:
        ancestors[target_node] = (current_path[:-1],current_path_name[:-1])  # Exclude the target node itself

    for child in node.children:
        _find_ancestors_helper(child, target_node, current_path, current_path_name, ancestors)

    current_path.pop()
    current_path_name.pop()

def find_leaf_all_parents(leaf_products_dict, cur_tree):
    ancestors = find_ancestors(cur_tree, list(leaf_products_dict.keys()))
    updated_ancestor_dict = {}
    for ancestor_key in ancestors.keys():
        if len(ancestors[ancestor_key]) == 0:
            continue
        else:
            updated_ancestor_dict[ancestor_key] = ancestors[ancestor_key][1]
    return updated_ancestor_dict

def find_nodes_at_level(root, level):
    result = []
    _find_nodes_at_level_helper(root, level, 1, result)
    return result


def _find_nodes_at_level_helper(node, target_level, current_level, result):
    if current_level == target_level:
        result.append(node)

    for child in node.children:
        _find_nodes_at_level_helper(child, target_level, current_level + 1, result)



def sample_nodes_hard_level(leaf_product_dict, leaf_parent_dict, nodes_at_level_n, nodes_at_level_n1, cur_tree_name, out_path, cur_level, question_mode = 0):
    exclusion = ['Books', 'Wine', 'Collectibles and Fine Arts', 'Handmade', 'Magazine Subscriptions', 'Apps and Games', 'Movies and TV', 'CDs and Vinyl', 'Digital Music', 'Amazon Instant Video', 'Clothing, Shoes and Jewelry', 'Kindle Store']
    partial_exclusion = ['Baby', 'Clothing, Shoes and Jewelry - Baby', 'Clothing, Shoes and Jewelry - Boys', 'Clothing, Shoes and Jewelry - Girls', 'Clothing, Shoes and Jewelry - Men', 'Clothing, Shoes and Jewelry - Women']

    total_leaf_size = 0
    cleaned_leaf_product_dict = {}
    for key in leaf_product_dict.keys():
        if len(leaf_product_dict[key])>0:
            total_leaf_size += 1
            cleaned_leaf_product_dict[key] = leaf_product_dict[key]
    
    nodes_names_at_level_n = [node.name for node in nodes_at_level_n]

    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 1: # negative hard question; 每个leaf sample一个product生成 question bank
            sampled_idx = range(total_leaf_size)
            negative_sampled_pairs = []
            for sampled_id in sampled_idx:
                cur_leaf = list(cleaned_leaf_product_dict.keys())[sampled_id]
                cur_ancestors = leaf_parent_dict[cur_leaf]
                num_parents = len(cur_ancestors)
                if cur_level > num_parents:
                    continue
                else:
                    cur_root = cur_tree_name
                    setup_seed(20)
                    sampled_pair = random.choice(cleaned_leaf_product_dict[cur_leaf])
                    cur_grand = cur_ancestors[cur_level-1]
                    for node in nodes_at_level_n1:
                        if node.name == cur_grand:
                            cur_grand_node = node
                            break
                    cur_root_set_nodes = cur_grand_node.children
                    cur_root_set = [node.name for node in cur_root_set_nodes]
                    if len(cur_root_set) == 0:
                        continue
                    random.seed(datetime.datetime.now())
                    cur_parent = random.choice(cur_root_set)
                    while cur_parent == cur_leaf:
                        cur_parent = random.choice(cur_root_set)
                    if (cur_root in exclusion) or (cur_root in partial_exclusion):
                        negative_sampled_pairs.append((cur_parent+ ' (' + cur_root + ')', sampled_pair))
                    else:
                        negative_sampled_pairs.append((cur_parent, sampled_pair))

            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('shopping-amazon', clean_name(root), clean_name(child), 'instance'+str(cur_level+1), 'level-negative-hard')
                csv_writer.writerow(cur_template)
            print('size of the negative set', len(negative_sampled_pairs))

        elif question_mode == 0: # positive
            
            sampled_idx = range(total_leaf_size)
            positive_sampled_pairs = []
            for sampled_id in sampled_idx:
                cur_leaf = list(cleaned_leaf_product_dict.keys())[sampled_id]
                cur_ancestors = leaf_parent_dict[cur_leaf]
                num_parents = len(cur_ancestors)
                if cur_level > num_parents:
                    continue
                elif cur_level == num_parents:
                    cur_root = cur_tree_name
                    setup_seed(20)
                    sampled_pair = random.choice(cleaned_leaf_product_dict[cur_leaf])
                    if (cur_root in exclusion) or (cur_root in partial_exclusion):
                        positive_sampled_pairs.append((cur_leaf+ ' (' + cur_root + ')', sampled_pair))
                    else:
                        positive_sampled_pairs.append((cur_leaf, sampled_pair))
                else:
                    cur_root = cur_tree_name
                    setup_seed(20)
                    sampled_pair = random.choice(cleaned_leaf_product_dict[cur_leaf])
                    cur_parent = cur_ancestors[cur_level]
                    if (cur_root in exclusion) or (cur_root in partial_exclusion):
                        positive_sampled_pairs.append((cur_parent+ ' (' + cur_root + ')', sampled_pair))
                    else:
                        positive_sampled_pairs.append((cur_parent, sampled_pair))

            for (root, child) in tqdm(positive_sampled_pairs):
                cur_template = ('shopping-amazon', clean_name(root), clean_name(child), 'instance'+str(cur_level+1), 'level-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(positive_sampled_pairs))
        
        elif question_mode == 2: # negative easy question; 每个leaf sample一个product生成 question bank
            sampled_idx = range(total_leaf_size)
            negative_sampled_pairs = []
            for sampled_id in sampled_idx:
                cur_leaf = list(cleaned_leaf_product_dict.keys())[sampled_id]
                cur_ancestors = leaf_parent_dict[cur_leaf]
                cur_root_set = nodes_names_at_level_n
                num_parents = len(cur_ancestors)
                if cur_level > num_parents:
                    continue
                else:
                    cur_root = cur_tree_name
                    setup_seed(20)
                    sampled_pair = random.choice(cleaned_leaf_product_dict[cur_leaf])
                    random.seed(datetime.datetime.now())
                    cur_parent = random.choice(cur_root_set)
                    while cur_parent == cur_leaf:
                        cur_parent = random.choice(cur_root_set)
                    if (cur_root in exclusion) or (cur_root in partial_exclusion):
                        negative_sampled_pairs.append((cur_parent+ ' (' + cur_root + ')', sampled_pair))
                    else:
                        negative_sampled_pairs.append((cur_parent, sampled_pair))

            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('shopping-amazon', clean_name(root), clean_name(child), 'instance'+str(cur_level+1), 'level-negative-easy')
                csv_writer.writerow(cur_template)
            print('size of the negative set', len(negative_sampled_pairs))

all_leaf_product_dicts = []
all_trees_names = []
all_leaf_sibling_mappings = []


file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv']


for q_idx, file in enumerate(file_name):
    out_path = 'TaxoGlimpse/question_pools/shopping-amazon/instance_full/question_pool_full_updated_'+file
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
        
    for n in range(2,5):
        nodes_at_level_ns = []
        for idx in trange(len(node_names)):
            with open('TaxoGlimpse/LLM-taxonomy/shopping/data/browsenodes-leaf-to-product/'+str(idx)+'.pkl', 'rb') as pickle_file:
                cur_leaf_product_dict = pickle.load(pickle_file)
            with open('TaxoGlimpse/LLM-taxonomy/shopping/data/browsenodes/'+str(idx)+'.pkl', 'rb') as pickle_file_tree:
                cur_tree = pickle.load(pickle_file_tree)
            nodes_at_level_n = find_nodes_at_level(cur_tree, n) # node object
            nodes_at_level_ns += nodes_at_level_n

        for idx in trange(len(node_names)):
            with open('TaxoGlimpse/LLM-taxonomy/shopping/data/browsenodes-leaf-to-product/'+str(idx)+'.pkl', 'rb') as pickle_file:
                cur_leaf_product_dict = pickle.load(pickle_file)
            with open('TaxoGlimpse/LLM-taxonomy/shopping/data/browsenodes/'+str(idx)+'.pkl', 'rb') as pickle_file_tree:
                cur_tree = pickle.load(pickle_file_tree)
            cur_tree_name = cur_tree.name

            cur_updated_ancestor_dict = find_leaf_all_parents(cur_leaf_product_dict, cur_tree)
            updated_cur_leaf_product_dict = {clean_name(key): cur_leaf_product_dict[key] for key in cur_leaf_product_dict.keys()}

            intersection_leaf_products = {key: updated_cur_leaf_product_dict[key] for key in cur_updated_ancestor_dict.keys() & updated_cur_leaf_product_dict.keys()}
            intersection_leaf_parents = {key: cur_updated_ancestor_dict[key] for key in cur_updated_ancestor_dict.keys() & updated_cur_leaf_product_dict.keys()}
            
            cur_level = n-1
            #nodes_at_level_n = find_nodes_at_level(cur_tree, n) # node object
            nodes_at_level_n1 = find_nodes_at_level(cur_tree, n-1) # node object

            cur_level_siblings = []
            #for node in nodes_at_level_n:
            #    cur_ancestors = find_ancestors(cur_tree, [node.name])
            #    cur_parent_node = list(cur_ancestors.values())[0][0][-1]
            #    cur_siblings_nodes = cur_parent_node.children
            #    cur_siblings = [sib.name for sib in cur_siblings_nodes]
            #    cur_siblings_exclude_self = list(set(cur_siblings)-set([node.name]))
            #    cur_level_siblings.append(cur_siblings_exclude_self)
               
            sample_nodes_hard_level(intersection_leaf_products, intersection_leaf_parents, nodes_at_level_ns, nodes_at_level_n1, cur_tree_name, out_path, cur_level, question_mode = q_idx)



    #leaf_sibling_mapping = bfs_traversal(cur_tree, list(cur_leaf_product_dict.keys()))
    #with open('TaxoGlimpse/LLM-taxonomy/shopping/data/browsenodes-leaf-siblings/'+str(idx)+'.pkl', 'wb') as pickle_file:
    #    pickle.dump(leaf_sibling_mapping, pickle_file)
    
    
    with open('TaxoGlimpse/LLM-taxonomy/shopping/data/browsenodes-leaf-siblings/'+str(idx)+'.pkl', 'rb') as pickle_file:
        leaf_sibling_mapping = pickle.load(pickle_file)
    
    all_leaf_product_dicts.append(cur_leaf_product_dict)
    all_trees_names.append(cur_tree_name)
    all_leaf_sibling_mappings.append(leaf_sibling_mapping)
    # 调用时记得用cur_tree_name做一下筛选 并且做clean name

file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv', 'positive_to_root.csv', 'negative_to_root.csv']
modes = [1,2,4]

for cur_question_mode in modes:
    out_path = 'TaxoGlimpse/question_pools/shopping-amazon/instance_typing/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    sample_nodes_hard(all_leaf_product_dicts, all_leaf_sibling_mappings, all_trees_names, out_path=out_path, question_mode=cur_question_mode)
