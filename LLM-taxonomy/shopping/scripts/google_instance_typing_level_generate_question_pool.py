import csv
import random
from tqdm import tqdm
import pickle
import os
import datetime


def load_crawled_dict():
    dict_base_path = 'TaxoGlimpse/LLM-taxonomy/shopping/data/crawled/'

    crawled_dict = {}
    for pickled_file_name in os.listdir(dict_base_path):
        if pickled_file_name == 'product_dict_13.pkl':
            pickled_file_path = dict_base_path + pickled_file_name
            with open(pickled_file_path, 'rb') as file:
                pickled_dict = dict(pickle.load(file))
            for key, value in pickled_dict.items():
                if not value:
                    continue
                else:
                    crawled_dict[key.split('finder/')[1]] = value
    return crawled_dict

def replace_and(string):
    return string.replace('&', 'and')

def replace_blank(string):
    return string.replace(' ','-')

def replace_line(string):
    return string.replace('-',' ')

def clean_each_level(current_tax_path):
    out_link = ''
    cleaned_tax_path = []
    for item in current_tax_path:
        cleaned_tax_path.append(replace_blank(replace_and(item.strip())).lower())
    for item in cleaned_tax_path:
        out_link += item
        out_link += '/'
    return out_link

def load_all_paths():
    txt_path = 'TaxoGlimpse/LLM-taxonomy/shopping/data/google-US.txt'
    cleaned_links = []
    with open(txt_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for idx, line in enumerate(content):
            if idx == 0:
                continue
            else:
                current_tax_path = line.strip().split('>')
                if len(current_tax_path) == 1:
                    continue
                else:
                    cleaned_links.append(clean_each_level(current_tax_path))
    return cleaned_links

def setup_seed(seed):
    random.seed(seed)

def get_level_n(all_dicts, n):
    level_set = []
    for item in all_dicts:
        cur_chain = item[:-1].split('/')
        if len(cur_chain) > n:
            level_set.append(cur_chain[n-1])
    return list(set(level_set))

def get_parent_kids(all_dicts, parent):
    kids = []
    for item in all_dicts:
        cur_chain = item[:-1].split('/')
        if parent in cur_chain:
            parent_idx = cur_chain.index(parent)
            if parent_idx+1 >= len(cur_chain):
                continue
            kids.append(cur_chain[parent_idx+1])
    return kids


def sample_nodes_hard_level(crawled_dict, all_dicts, out_path, cur_level, question_mode = 0):
    total_leaf_size = len(list(crawled_dict.keys()))
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 1: # negative hard question; 每个leaf sample一个product生成 question bank
            sampled_idx = range(total_leaf_size)
            negative_sampled_pairs = []
            parents_level_n = get_level_n(all_dicts=all_dicts, n=cur_level)
            for sampled_id in sampled_idx:
                cur_products = crawled_dict[list(crawled_dict.keys())[sampled_id]]
                cur_ancestors = list(crawled_dict.keys())[sampled_id][:-1].split('/')
                num_parents = len(cur_ancestors)
                if cur_level > num_parents:
                    continue
                else:
                    if cur_level == 1:
                        random.seed(datetime.datetime.now())
                        cur_parent = random.choice(list(set(parents_level_n)-set([cur_ancestors[0]])))
                    else:
                        cur_grand = cur_ancestors[cur_level-2]
                        cur_candidates = get_parent_kids(all_dicts, cur_grand)
                        if len(cur_candidates) <= 1:
                            continue
                        else:
                            random.seed(datetime.datetime.now())
                            cur_parent = random.choice(list(set(cur_candidates)-set([cur_ancestors[cur_level-2]])))
                    setup_seed(20)
                    cur_child = random.choice(cur_products)
                    negative_sampled_pairs.append((cur_parent, cur_child))

            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('shopping-google', replace_and(root), replace_and(child), 'instance'+str(cur_level), 'level-negative-hard')
                csv_writer.writerow(cur_template)
            print('size of the negative set', len(negative_sampled_pairs))

        elif question_mode == 0: # positive
            
            sampled_idx = range(total_leaf_size)
            positive_sampled_pairs = []
            for sampled_id in sampled_idx:
                cur_products = crawled_dict[list(crawled_dict.keys())[sampled_id]]
                cur_ancestors = list(crawled_dict.keys())[sampled_id][:-1].split('/')
                num_parents = len(cur_ancestors)
                if cur_level > num_parents:
                    continue
                else:
                    cur_parent = cur_ancestors[cur_level-1]
                    setup_seed(20)
                    cur_child = random.choice(cur_products)
                    positive_sampled_pairs.append((cur_parent, cur_child))

            for (root, child) in tqdm(positive_sampled_pairs):
                cur_template = ('shopping-google', replace_and(root), replace_and(child), 'instance'+str(cur_level), 'level-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(positive_sampled_pairs))
        
        elif question_mode == 2: # negative easy question; 每个leaf sample一个product生成 question bank
            sampled_idx = range(total_leaf_size)
            negative_sampled_pairs = []
            parents_level_n = get_level_n(all_dicts=all_dicts, n=cur_level)
            for sampled_id in sampled_idx:
                cur_products = crawled_dict[list(crawled_dict.keys())[sampled_id]]
                cur_ancestors = list(crawled_dict.keys())[sampled_id][:-1].split('/')
                num_parents = len(cur_ancestors)
                if cur_level > num_parents:
                    continue
                else:
                    random.seed(datetime.datetime.now())
                    cur_parent = random.choice(list(set(parents_level_n)-set([cur_ancestors[cur_level-1]])))
                    setup_seed(20)
                    cur_child = random.choice(cur_products)
                    negative_sampled_pairs.append((cur_parent, cur_child))

            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('shopping-google', replace_and(root), replace_and(child), 'instance'+str(cur_level), 'level-negative-easy')
                csv_writer.writerow(cur_template)
            print('size of the negative set', len(negative_sampled_pairs))


file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv']
crawled_dict = load_crawled_dict()
all_dicts = load_all_paths()

for q_idx, file in enumerate(file_name):
    out_path = 'TaxoGlimpse/question_pools/shopping-google/instance_full/question_pool_full_'+file
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    for n in range(1,5):
        sample_nodes_hard_level(crawled_dict, all_dicts, out_path, cur_level=n, question_mode = q_idx)
    