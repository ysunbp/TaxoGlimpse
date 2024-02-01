import simple_icd_10_cm as cm
import random
from tqdm import tqdm
import time
import csv



def get_second_level():
    all_codes = cm.get_all_codes()

    top_level = [] # 22 types
    top_level_names = []
    second_level = [] # 258 types
    second_level_names = []
    for code in all_codes:
        if code.isdigit():
            top_level.append(code) # root level code for classification
            top_level_names.append(cm.get_description(code).split('(')[0]) # root level classification description
        elif "-" in code:
            second_level.append(code)
            second_level_names.append(cm.get_description(code).split('(')[0])
    return second_level
    
def get_cur_level_pairs(second_level, level):
    cur_pairs = []
    for second_type in tqdm(second_level):
        if level == 1:
            typeA = cm.get_description(cm.get_ancestors(second_type)[0]).split('(')[0].lower()
            typeB = cm.get_description(second_type).split('(')[0].lower()
            if ('elsewhere' in typeA) or ('other' in typeA) or ('elsewhere' in typeB) or ('other' in typeB) or ('unspecified' in typeA) or ('unspecified' in typeB) or (typeA in typeB) or (typeB in typeA):
                continue
            else:   
                cur_pairs.append((typeA.strip(), typeB.strip()))
        else:
            cur_children = cm.get_children(second_type)
            if level == 2:
                typeA = cm.get_description(second_type).split('(')[0].lower()
                for cur_child in cur_children:
                    typeB = cm.get_description(cur_child).lower()
                    if ('elsewhere' in typeA) or ('other' in typeA) or ('elsewhere' in typeB) or ('other' in typeB) or ('unspecified' in typeA) or ('unspecified' in typeB) or (typeA in typeB) or (typeB in typeA):
                        continue
                    else:   
                        cur_pairs.append((typeA.strip(), typeB.strip()))
            else:
                for cur_child in cur_children:
                    typeA = cm.get_description(cur_child).lower()
                    cur_grand_children = cm.get_children(cur_child)
                    for cur_grand_child in cur_grand_children:
                        typeB = cm.get_description(cur_grand_child).lower()
                        if ('elsewhere' in typeA) or ('other' in typeA) or ('elsewhere' in typeB) or ('other' in typeB) or ('unspecified' in typeA) or ('unspecified' in typeB) or (typeA in typeB) or (typeB in typeA):
                            continue
                        else:   
                            cur_pairs.append((typeA.strip(), typeB.strip()))
    return cur_pairs

def get_cur_level_pairs_hard(second_level, level):
    cur_pairs = []
    nodes_uncles_dict = {}
    top_list = ['13', '6', '4', '10', '7', '8', '11', '17', '15', '22', '3', '20', '18', '19', '2', '21', '12', '14', '9', '1', '5', '16'] # top level classification codes of ICD-10-CM
    for second_type in tqdm(second_level):
        if level == 1:
            typeA = cm.get_description(cm.get_ancestors(second_type)[0]).split('(')[0].lower()
            top_list.append(cm.get_ancestors(second_type)[0])
            typeB = cm.get_description(second_type).split('(')[0].lower()
            if ('elsewhere' in typeA) or ('other' in typeA) or ('elsewhere' in typeB) or ('other' in typeB) or ('unspecified' in typeA) or ('unspecified' in typeB) or (typeA in typeB) or (typeB in typeA):
                continue
            else:   
                cur_pairs.append((typeA.strip(), typeB.strip()))
                #print(list(set(top_list)-set([cm.get_ancestors(second_type)[0]])))
                nodes_uncles_dict[typeB.strip()] = list(set(top_list)-set([cm.get_ancestors(second_type)[0]]))
            
        else:
            cur_children = cm.get_children(second_type)
            if level == 2:
                typeA = cm.get_description(second_type).split('(')[0].lower()
                
                for cur_child in cur_children:
                    typeB = cm.get_description(cur_child).lower()                        
                    if ('elsewhere' in typeA) or ('other' in typeA) or ('elsewhere' in typeB) or ('other' in typeB) or ('unspecified' in typeA) or ('unspecified' in typeB) or (typeA in typeB) or (typeB in typeA):
                        continue
                    else:   
                        cur_pairs.append((typeA.strip(), typeB.strip()))
                        nodes_uncles_dict[typeB.strip()] = list(set(cm.get_children(cm.get_ancestors(second_type)[0]))-set([second_type]))
            
            else:
                for cur_child in cur_children:
                    typeA = cm.get_description(cur_child).lower()
                    cur_grand_children = cm.get_children(cur_child)
                    for cur_grand_child in cur_grand_children:
                        typeB = cm.get_description(cur_grand_child).lower()
                        if ('elsewhere' in typeA) or ('other' in typeA) or ('elsewhere' in typeB) or ('other' in typeB) or ('unspecified' in typeA) or ('unspecified' in typeB) or (typeA in typeB) or (typeB in typeA):
                            continue
                        else:   
                            cur_pairs.append((typeA.strip(), typeB.strip()))
                            nodes_uncles_dict[typeB.strip()] = list(set(cur_children)-set([cur_child]))
    print('total number of pairs', len(cur_pairs))
    return cur_pairs, nodes_uncles_dict


def get_cur_level_pairs_root(second_level, level):
    cur_pairs = []
    nodes_uncles_dict = {}
    top_list = ['13', '6', '4', '10', '7', '8', '11', '17', '15', '22', '3', '20', '18', '19', '2', '21', '12', '14', '9', '1', '5', '16'] # top level classification codes of ICD-10-CM
    for second_type in tqdm(second_level):
        if level == 1:
            typeA = cm.get_description(cm.get_ancestors(second_type)[0]).split('(')[0].lower()
            top_list.append(cm.get_ancestors(second_type)[0])
            typeB = cm.get_description(second_type).split('(')[0].lower()
            if ('elsewhere' in typeA) or ('other' in typeA) or ('elsewhere' in typeB) or ('other' in typeB) or ('unspecified' in typeA) or ('unspecified' in typeB) or (typeA in typeB) or (typeB in typeA):
                continue
            else:   
                cur_pairs.append((typeA.strip(), typeB.strip()))
                #print(list(set(top_list)-set([cm.get_ancestors(second_type)[0]])))
                nodes_uncles_dict[typeB.strip()] = list(set(top_list)-set([cm.get_ancestors(second_type)[0]]))
            
        else:
            cur_children = cm.get_children(second_type)
            if level == 2:
                typeA = cm.get_description(cm.get_ancestors(second_type)[0]).split('(')[0].lower()
                
                for cur_child in cur_children:
                    typeB = cm.get_description(cur_child).lower()                        
                    if ('elsewhere' in typeA) or ('other' in typeA) or ('elsewhere' in typeB) or ('other' in typeB) or ('unspecified' in typeA) or ('unspecified' in typeB) or (typeA in typeB) or (typeB in typeA):
                        continue
                    else:   
                        cur_pairs.append((typeA.strip(), typeB.strip()))
                        nodes_uncles_dict[typeB.strip()] = list(set(cm.get_children(cm.get_ancestors(second_type)[0]))-set([second_type]))
            
            else:
                for cur_child in cur_children:
                    typeA = cm.get_description(cm.get_ancestors(second_type)[0]).split('(')[0].lower()
                    cur_grand_children = cm.get_children(cur_child)
                    for cur_grand_child in cur_grand_children:
                        typeB = cm.get_description(cur_grand_child).lower()
                        if ('elsewhere' in typeA) or ('other' in typeA) or ('elsewhere' in typeB) or ('other' in typeB) or ('unspecified' in typeA) or ('unspecified' in typeB) or (typeA in typeB) or (typeB in typeA):
                            continue
                        else:   
                            cur_pairs.append((typeA.strip(), typeB.strip()))
                            nodes_uncles_dict[typeB.strip()] = list(set(cur_children)-set([cur_child]))
    print('total number of pairs', len(cur_pairs))
    return cur_pairs, nodes_uncles_dict

def sample_question_pool(cur_level_pairs, nodes_uncles_dict, sample_size, cur_level, out_path, question_mode=0):
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 1: # negative hard question
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
            for idx, sampled_pair in enumerate(sampled_pairs):
                cur_root_set_code = nodes_uncles_dict[sampled_pair[1]]
                cur_root_set = []
                typeB = sampled_pair[1]
                for cur_root_code in cur_root_set_code:
                    typeA = cm.get_description(cur_root_code).split('(')[0].lower()
                    if ('elsewhere' in typeA) or ('other' in typeA) or ('unspecified' in typeA) or (typeA in typeB) or (typeB in typeA):
                        continue
                    else:
                        cur_root_set.append(typeA)
                if len(cur_root_set) == 0:
                    continue
                    
                cur_negative_parent = random.choice(cur_root_set)
                negative_sampled_pairs.append((cur_negative_parent,sampled_pair[1]))
            
            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('medical-icd', root, child, cur_level, 'level-negative-hard')
                csv_writer.writerow(cur_template)
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        elif question_mode == 0:
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            for (root, child) in tqdm(sampled_pairs):
                cur_template = ('medical-icd', root, child, cur_level, 'level-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_pairs), cur_level)
        elif question_mode == 2:
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
            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('medical-icd', root, child, cur_level, 'level-negative-easy')
                csv_writer.writerow(cur_template)
            print('size of the negative set', len(negative_sampled_pairs), cur_level)
        elif question_mode == 3:
            if len(cur_level_pairs) < sample_size:
                sampled_pairs = cur_level_pairs
            else:
                sampled_pairs = random.sample(cur_level_pairs, sample_size)
            for (root, child) in tqdm(sampled_pairs):
                cur_template = ('medical-icd', root, child, cur_level, 'toroot-positive')
                csv_writer.writerow(cur_template)
            print('size of the positive set', len(sampled_pairs), cur_level)
        elif question_mode == 4:
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
            for idx, sampled_pair in enumerate(sampled_pairs):
                cur_root_set_code = nodes_uncles_dict[sampled_pair[1]]
                cur_root_set = []
                typeB = sampled_pair[1]
                for cur_root_code in cur_root_set_code:
                    typeA = cm.get_description(cur_root_code).split('(')[0].lower()
                    if ('elsewhere' in typeA) or ('other' in typeA) or ('unspecified' in typeA) or (typeA in typeB) or (typeB in typeA):
                        continue
                    else:
                        cur_root_set.append(typeA)
                if len(cur_root_set) == 0:
                    continue
                    
                cur_negative_parent = random.choice(cur_root_set)
                negative_sampled_pairs.append((cur_negative_parent,sampled_pair[1]))
            
            for (root, child) in tqdm(negative_sampled_pairs):
                cur_template = ('medical-icd', root, child, cur_level, 'toroot-negative-hard')
                csv_writer.writerow(cur_template)
            print('size of the negative set', len(negative_sampled_pairs), cur_level)

def setup_seed(seed):
    random.seed(seed)

seed = 20
setup_seed(seed)
print('current seed', seed)

num_question_samples = [100000, 100000, 100000]

file_name = ['positive.csv', 'negative_hard.csv', 'negative_easy.csv', 'positive_to_root.csv', 'negative_to_root.csv']


for cur_question_mode in range(3):
    out_path = 'TaxoGlimpse/question_pools/medical-icd/level/level_question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    
    for cur_level in range(1, len(num_question_samples)+1):
        if cur_question_mode < 3:
            cur_level_pairs, nodes_uncles_dict = get_cur_level_pairs_hard(get_second_level(), level=cur_level) # level 1是从level 1 到root
        else:    
            cur_level_pairs = get_cur_level_pairs(get_second_level(), level=cur_level)
            nodes_uncles_dict = 0
        sample_question_pool(cur_level_pairs, nodes_uncles_dict, sample_size=num_question_samples[cur_level-1], cur_level=cur_level, out_path=out_path, question_mode=cur_question_mode)


for cur_question_mode in range(3,5):
    out_path = 'TaxoGlimpse/question_pools/medical-icd/toroot/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type'])  # 写入表头
    
    for cur_level in range(1, len(num_question_samples)+1):
        cur_level_pairs, nodes_uncles_dict = get_cur_level_pairs_root(get_second_level(), level=cur_level) # level 1是从level 1 到root
        sample_question_pool(cur_level_pairs, nodes_uncles_dict, sample_size=num_question_samples[cur_level-1], cur_level=cur_level, out_path=out_path, question_mode=cur_question_mode)
