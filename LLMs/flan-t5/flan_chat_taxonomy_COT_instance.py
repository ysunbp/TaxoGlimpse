import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import csv
import random
from tqdm import tqdm

### copied from evaluate_llama_taxonomy.py
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

def sample_question_pairs(question_pool_dict):
    sampled_question_pairs = {}
    for question_pool_key in question_pool_dict.keys():
        sample_size = compute_cur_level_size(len(question_pool_dict[question_pool_key]))
        setup_seed(20)
        sampled_question_pairs[question_pool_key] = random.sample(question_pool_dict[question_pool_key], sample_size)
    return sampled_question_pairs

def get_sampled_pairs(sub_question_type='level', level_question_types=['positive'], toroot_question_types=None, question_pool_name = 'academic-acm'):
    question_pools_base = 'TaxoGlimpse/question_pools/'
    cur_question_pool = question_pools_base+question_pool_name+'/'
    
    if sub_question_type == 'level':
        sampled_question_pairs = {}
        for level_question_type in level_question_types:
            cur_question_pool_path = cur_question_pool+'instance_typing/'
            cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
            cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
            sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
    
    return sampled_question_pairs

### copied from evaluate_llama_taxonomy.py

def move_tensors_to_gpu(data):
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to('cuda')
    return data

def compose_questions_get_response(sampled_question_pairs, taxonomy_type, model, tokenizer, model_name='flan-t5-3b'):
    system_message = "Always answer with Yes, No, or I don't know. "
    cot_message = "Let's think step by step."
    print('current taxonomy type:', taxonomy_type)
    for sampled_question_key in sampled_question_pairs.keys():
        cur_question_cat = sampled_question_key
        total_num_questions = len(sampled_question_pairs[sampled_question_key])
        cur_path = 'TaxoGlimpse/results/instance-COT/'+taxonomy_type+'/'+model_name+'/'+cur_question_cat+'.csv'
        yes_total = 0
        no_total = 0
        dont_total = 0
        with open(cur_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            for (roottype, subtype) in tqdm(sampled_question_pairs[sampled_question_key]):
                if taxonomy_type == 'academic-acm':
                    dialog = system_message+'Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)' + cot_message+'\nAnswer:'
                elif taxonomy_type == 'biology-NCBI':
                    dialog = system_message+'Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)' + cot_message+'\nAnswer:'
                elif taxonomy_type == 'language-glottolog':
                    dialog = system_message+'Is ' + subtype + ' language a type of '+roottype +' language? answer with (Yes/No/I don\'t know)' + cot_message+'\nAnswer:'
                elif taxonomy_type == 'medical-icd':
                    dialog = system_message+'Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)' + cot_message+'\nAnswer:'
                elif taxonomy_type == 'shopping-amazon':
                    dialog = system_message+'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)' + cot_message+'\nAnswer:'
                elif taxonomy_type == 'shopping-google':
                    dialog = system_message+'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)' + cot_message+'\nAnswer:'
                
                inputs = tokenizer(dialog, return_tensors="pt")
                inputs = move_tensors_to_gpu(inputs)
                outputs = model.generate(**inputs, max_length=256)
                #print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
                flan_current_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip().lower()
                #print(dialog['user'])
                #print(llama_current_answer)
                #print('=======================================')
                
                if 'yes' in flan_current_answer:
                    decision = 1
                    yes_total += 1
                elif 'know' in flan_current_answer:
                    decision = 2
                    dont_total += 1
                else:
                    decision = 0
                    no_total += 1
                cur_row = (roottype, subtype, flan_current_answer, decision)
                csv_writer.writerow(cur_row)

        if 'positive' in cur_question_cat:
            acc = yes_total/total_num_questions
        else:
            acc = no_total/total_num_questions
        miss_rate = dont_total/total_num_questions
        print('summary of exp:', cur_question_cat, ', total number of questions:', total_num_questions, ', yes total:', yes_total, ', no total:', no_total, ', miss total:', dont_total, 'acc', acc, 'missing rate', miss_rate)
        print('++++++++++++++++++++++++++++++++++++++++')
    

if __name__ == "__main__":
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").cuda() # 3B
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    question_pool_names = ['shopping-amazon']

    for question_pool_name in question_pool_names:
        cur_question_pairs = []
        sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy', "positive_to_root", "negative_to_root"], toroot_question_types=None, question_pool_name = question_pool_name)
        cur_question_pairs.append(sampled_question_pairs)
        for cur_question_pair in cur_question_pairs:
            compose_questions_get_response(cur_question_pair, taxonomy_type=question_pool_name, model=model, tokenizer=tokenizer, model_name='flan-t5-3b')

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl").cuda() # 11B
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    question_pool_names = ['shopping-amazon']

    for question_pool_name in question_pool_names:
        cur_question_pairs = []
        sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy', "positive_to_root", "negative_to_root"], toroot_question_types=None, question_pool_name = question_pool_name)
        cur_question_pairs.append(sampled_question_pairs)
        for cur_question_pair in cur_question_pairs:
            compose_questions_get_response(cur_question_pair, taxonomy_type=question_pool_name, model=model, tokenizer=tokenizer, model_name='flan-t5-11b')
