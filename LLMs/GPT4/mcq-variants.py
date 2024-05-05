import csv
import random
from tqdm import tqdm
from openai import OpenAI

import time
import os

client = OpenAI(
    base_url = 'xxxxxx',
    api_key = 'xxxxxx'
)

def generateResponse(prompt, gpt_name, exp_type, client):
    
    '''
    Input:
        prompt (str): composed prompt.
    Output:
        response (str): response from GPT-4.

    This function takes composed prompt as input and returns the response from GPT.
    '''
    if exp_type == 'mcq':
        messages = [{"role": "system", "content": prompt['system']}, {"role": "user", "content": prompt['user']}]

    response = client.chat.completions.create(
        model=gpt_name,
        temperature=0,
        messages=messages
    )
    return response.choices[0].message.content

def compute_cur_level_size(total_samples):
    # 95% confidence, margin of error 5%
    return int(((1.96**2*0.5*(1-0.5)/0.05**2)/(1+(1.96**2*0.5*(1-0.5)/(0.05**2*total_samples))))+1)

def setup_seed(seed):
    random.seed(seed)

def load_csv_file(csv_path, question_type):
    question_pools = {}
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for idx, row in enumerate(csvreader):
            if idx == 0:
                continue
            else:
                cur_level = 'level_'+row[3]+'_'+question_type
                cur_parent = row[1]
                cur_child = row[2]
                if not cur_level in question_pools.keys():
                    question_pools[cur_level] = [(cur_parent, cur_child)]
                else:
                    question_pools[cur_level].append((cur_parent, cur_child))
    return question_pools

def load_csv_file_mcq(csv_path, question_type): # MODIFIED
    question_pools = {}
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for idx, row in enumerate(csvreader):
            if idx == 0:
                continue
            else:
                cur_level = 'mcq_'+row[3]+'_'+question_type
                cur_parent = row[1]
                cur_child = row[2]
                cur_alternatives = [row[5], row[6], row[7]]
                if not cur_level in question_pools.keys():
                    question_pools[cur_level] = [(cur_parent, cur_child, cur_alternatives)]
                else:
                    question_pools[cur_level].append((cur_parent, cur_child, cur_alternatives))
    return question_pools

def sample_question_pairs(question_pool_dict):
    sampled_question_pairs = {}
    for question_pool_key in question_pool_dict.keys():
        sample_size = compute_cur_level_size(len(question_pool_dict[question_pool_key]))
        setup_seed(20)
        sampled_question_pairs[question_pool_key] = random.sample(question_pool_dict[question_pool_key], sample_size)
    return sampled_question_pairs

def check_dup(sampled_list, item):
    flag = False
    for sampled_item in sampled_list:
        if sampled_item[1] == item[1]:
            flag = True
    return flag



def get_sampled_pairs(sub_question_type='level', level_question_types=['positive'], toroot_question_types=None, question_pool_name = 'academic-acm', exp_type = 'zero-shot'):
    question_pool_levels = {'acm':4, 'ncbi':6, 'glottolog':5, 'icd':3, 'amazon':4, 'google':4}
    question_pools_base = 'TaxoGlimpse/question_pools/'
    cur_question_pool = question_pools_base+question_pool_name+'/'

    cur_question_pool_path = cur_question_pool+sub_question_type+'/'
    if exp_type == 'mcq': # MODIFIED
        if not question_pool_name == 'biology-NCBI':
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                cur_csv_file = cur_question_pool_path + 'question_pool_full_mcq_'+level_question_type+'.csv'
                cur_question_pool_dict = load_csv_file_mcq(cur_csv_file, level_question_type)
                sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
        else:
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                for cur_level in range(1, question_pool_levels['ncbi']+1):
                    cur_csv_file = cur_question_pool_path + 'question_pool_full_level_' + str(cur_level) + '_mcq_' + level_question_type + '.csv'
                    cur_question_pool_dict = load_csv_file_mcq(cur_csv_file, level_question_type)
                    sampled_cur_level_question_pairs = sample_question_pairs(cur_question_pool_dict)
                    sampled_question_pairs.update(sampled_cur_level_question_pairs)

        return sampled_question_pairs
    

def shuffle_and_mark_answer(answer_list, roottype):
    # 随机排序答案列表
    random.shuffle(answer_list)
    
    # 找到正确答案的索引
    correct_index = answer_list.index(roottype)
    
    return answer_list, correct_index

def compose_question_templates_mcq(taxonomy_type, subtype, roottype, alternatives, variant_id=0): # MODIFIED
    if taxonomy_type == 'medical-icd':
        roottype = roottype.lower()
        subtype = subtype.lower()
        alternatives = [alternative.lower() for alternative in alternatives]
    all_choices = [roottype]+alternatives
    answer_list, correct_index = shuffle_and_mark_answer(all_choices, roottype)
    answer_notes = ['A)', 'B)', 'C)', 'D)']
    if taxonomy_type == 'academic-acm':
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+' research concept?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+' research concept?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+' research concept?' +'\n'
    elif taxonomy_type == 'biology-NCBI':
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+'?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+'?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+'?' +'\n'
    elif taxonomy_type == 'language-glottolog':
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+' language?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+' language?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+' language?' +'\n'
    elif taxonomy_type == 'medical-icd':
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+'?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+'?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+'?' +'\n'
    elif taxonomy_type == 'shopping-amazon':
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+' product?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+' product?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+' product?' +'\n'
    elif taxonomy_type == 'shopping-google':
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+' product?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+' product?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+' product?' +'\n'
    elif taxonomy_type == 'shopping-ebay':
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+' product?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+' product?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+' product?' +'\n'
    elif taxonomy_type == 'general-schema': 
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+' entity type?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+' entity type?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+' entity type?' +'\n'
    elif taxonomy_type == 'geography-geonames': 
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+' geographical concept?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+' geographical concept?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+' geographical concept?' +'\n'
    elif taxonomy_type == 'medical-oae':
        if variant_id == 0:
            question = 'What is the most appropriate supertype of '+subtype+' Adverse Events concept?' +'\n'
        elif variant_id == 1:
            question = 'What is the most suitable supertype of '+subtype+' Adverse Events concept?' +'\n'
        elif variant_id == 2:
            question = 'What is the most proper supertype of '+subtype+' Adverse Events concept?' +'\n'
    
    for answer_idx in range(4):
        question += answer_notes[answer_idx]
        question += answer_list[answer_idx]
        question += ' '

    return question, correct_index


def compose_questions_get_response_ctd(sampled_question_pairs, taxonomy_type, ckpt_dir, exp_type='zero-shot', variant_id=0):
    answer_notes = ['A)', 'B)', 'C)', 'D)']
    if exp_type == 'mcq':
        system_message = "Always answer with brief answers A), B), C), D), or I don't know."
        dialog = {}
        dialog['system'] = system_message
        print('current taxonomy type:', taxonomy_type, 'current variant id', variant_id)
        for sampled_question_key in sampled_question_pairs.keys():
            cur_question_cat = sampled_question_key
            total_num_questions = len(sampled_question_pairs[sampled_question_key])
            #cur_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'.csv'
            model_name = ckpt_dir.split('/')[0]
            
            if not variant_id == 0:
                folder_path = 'TaxoGlimpse/results/'+exp_type+'-'+str(variant_id)+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                cur_path = 'TaxoGlimpse/results/'+exp_type+'-'+str(variant_id)+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'.csv'
            else:
                folder_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                cur_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'.csv'
            yes_total = 0
            no_total = 0
            dont_total = 0
            with open(cur_path, 'a', newline='') as file:
                csv_writer = csv.writer(file)
                for (roottype, subtype, alternatives) in tqdm(sampled_question_pairs[sampled_question_key]):
                    dialog['user'], ground_truth_idx = compose_question_templates_mcq(taxonomy_type, subtype, roottype, alternatives, variant_id)
                    correct_ground_truth = answer_notes[ground_truth_idx].lower()
                    retry = 0
                    while True:
                        try:
                            gpt_current_answer = generateResponse(dialog, ckpt_dir, exp_type, client).lower()
                            time.sleep(1)
                            break
                        except Exception as error:
                            time.sleep(1)
                            retry += 1
                            if retry > 10:
                                gpt_current_answer = 'i don\'t know'
                                break

                    if correct_ground_truth in gpt_current_answer or correct_ground_truth[0] == gpt_current_answer[0]:
                        decision = 1
                        yes_total += 1
                    elif 'know' in gpt_current_answer:
                        decision = 2
                        dont_total += 1
                    else:
                        decision = 0
                        no_total += 1
                    
                    cur_row = (roottype, subtype, dialog['user'], correct_ground_truth, gpt_current_answer, decision)
                    csv_writer.writerow(cur_row)
            
            acc = yes_total/total_num_questions

            miss_rate = dont_total/total_num_questions
            print('summary of exp:', cur_question_cat, ', total number of questions:', total_num_questions, ', yes total:', yes_total, ', no total:', no_total, ', miss total:', dont_total, 'acc', acc, 'missing rate', miss_rate)
            print('++++++++++++++++++++++++++++++++++++++++')
    
if __name__ == "__main__":
    
    GPT_name = "gpt-4-1106-preview"

    question_pool_names = ['shopping-ebay', 'general-schema', 'geography-geonames', 'medical-oae', 'academic-acm', 'biology-NCBI', 'language-glottolog', 'medical-icd', 'shopping-amazon', 'shopping-google']
    variant_ids = [0,1,2]
    exp_type = 'mcq'
    
    if exp_type == 'mcq':  
        for question_pool_name in question_pool_names:
            for variant_id in variant_ids:
                cur_question_pairs = []
                
                sampled_question_pairs = get_sampled_pairs(sub_question_type='mcq', level_question_types=['hard'], toroot_question_types=None, question_pool_name = question_pool_name, exp_type=exp_type)
                cur_question_pairs.append(sampled_question_pairs)
                
                for cur_question_pair in cur_question_pairs:
                    compose_questions_get_response_ctd(cur_question_pair, taxonomy_type=question_pool_name, ckpt_dir = GPT_name, exp_type=exp_type, variant_id=variant_id) # GPU29
    
    
