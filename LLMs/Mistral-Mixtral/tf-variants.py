import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import random
from tqdm import tqdm

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

def compose_question_templates(taxonomy_type, subtype, roottype, variant_id=0):
    if taxonomy_type == 'academic-acm':
        if variant_id == 0:
            question = 'Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Is ' + subtype + ' computer science research concept a kind of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Is ' + subtype + ' computer science research concept a sort of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)'
    elif taxonomy_type == 'biology-NCBI':
        if variant_id == 0:
            question = 'Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Is ' + subtype + ' a kind of '+roottype +'? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Is ' + subtype + ' a sort of '+roottype +'? answer with (Yes/No/I don\'t know)'
    elif taxonomy_type == 'language-glottolog':
        if variant_id == 0:
            question = 'Is ' + subtype + ' language a type of '+roottype +' language? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Is ' + subtype + ' language a kind of '+roottype +' language? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Is ' + subtype + ' language a sort of '+roottype +' language? answer with (Yes/No/I don\'t know)'
    elif taxonomy_type == 'medical-icd':
        if variant_id == 0:
            question = 'Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Is ' + subtype.lower() + ' a kind of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Is ' + subtype.lower() + ' a sort of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)'
    elif taxonomy_type == 'shopping-amazon':
        if variant_id == 0:
            question = 'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Are ' + subtype + ' products a kind of '+roottype +' products? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Are ' + subtype + ' products a sort of '+roottype +' products? answer with (Yes/No/I don\'t know)'
    elif taxonomy_type == 'shopping-google':
        if variant_id == 0:
            question = 'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Are ' + subtype + ' products a kind of '+roottype +' products? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Are ' + subtype + ' products a sort of '+roottype +' products? answer with (Yes/No/I don\'t know)'
    elif taxonomy_type == 'shopping-ebay':
        if variant_id == 0:
            question = 'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Are ' + subtype + ' products a kind of '+roottype +' products? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Are ' + subtype + ' products a sort of '+roottype +' products? answer with (Yes/No/I don\'t know)'
    elif taxonomy_type == 'general-schema': 
        if variant_id == 0:
            question = 'Is ' + subtype + ' entity type a type of '+roottype +' entity type? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Is ' + subtype + ' entity type a kind of '+roottype +' entity type? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Is ' + subtype + ' entity type a sort of '+roottype +' entity type? answer with (Yes/No/I don\'t know)'
    elif taxonomy_type == 'geography-geonames': 
        if variant_id == 0:
            question = 'Is ' + subtype + ' geographical concept a type of '+roottype +' geographical concept? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Is ' + subtype + ' geographical concept a kind of '+roottype +' geographical concept? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Is ' + subtype + ' geographical concept a sort of '+roottype +' geographical concept? answer with (Yes/No/I don\'t know)'
    elif taxonomy_type == 'medical-oae':
        if variant_id == 0:
            question = 'Is '+ subtype + ' Adverse Events concept a type of '+roottype +' Adverse Events concept? answer with (Yes/No/I don\'t know)'
        elif variant_id == 1:
            question = 'Is '+ subtype + ' Adverse Events concept a kind of '+roottype +' Adverse Events concept? answer with (Yes/No/I don\'t know)'
        elif variant_id == 2:
            question = 'Is '+ subtype + ' Adverse Events concept a sort of '+roottype +' Adverse Events concept? answer with (Yes/No/I don\'t know)'
    return question



def sample_question_pairs(question_pool_dict):
    sampled_question_pairs = {}
    for question_pool_key in question_pool_dict.keys():
        sample_size = compute_cur_level_size(len(question_pool_dict[question_pool_key]))
        setup_seed(20)
        sampled_question_pairs[question_pool_key] = random.sample(question_pool_dict[question_pool_key], sample_size)
    return sampled_question_pairs

def get_sampled_pairs(sub_question_type='level', level_question_types=['positive'], toroot_question_types=None, question_pool_name = 'academic-acm'):
    question_pool_levels = {'acm':4, 'ncbi':6, 'glottolog':5, 'icd':3, 'amazon':4, 'google':4}
    question_pools_base = 'TaxoGlimpse/question_pools/'
    cur_question_pool = question_pools_base+question_pool_name+'/'

    cur_question_pool_path = cur_question_pool+sub_question_type+'/'
    if not question_pool_name == 'biology-NCBI':
        if sub_question_type == 'level':
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                cur_csv_file = cur_question_pool_path + 'level_question_pool_full_' + level_question_type + '.csv'
                cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                
        else:
            sampled_question_pairs = {}
            for toroot_question_type in toroot_question_types:
                cur_csv_file = cur_question_pool_path + 'question_pool_full_' + toroot_question_type + '.csv'
                cur_question_pool_dict = load_csv_file(cur_csv_file, toroot_question_type)
                sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
    else:
        if sub_question_type == 'level':
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                for cur_level in range(1, question_pool_levels['ncbi']+1):
                    cur_csv_file = cur_question_pool_path + 'level_question_pool_full_level_' + str(cur_level) + '_' + level_question_type + '.csv'
                    cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                    sampled_cur_level_question_pairs = sample_question_pairs(cur_question_pool_dict)
                    sampled_question_pairs.update(sampled_cur_level_question_pairs)
                
        else:
            sampled_question_pairs = {}
            for toroot_question_type in toroot_question_types:
                for cur_level in range(1, question_pool_levels['ncbi']+1):
                    cur_csv_file = cur_question_pool_path + 'question_pool_full_level_' + str(cur_level) + '_' + toroot_question_type + '.csv'
                    cur_question_pool_dict = load_csv_file(cur_csv_file, toroot_question_type)
                    sampled_cur_level_question_pairs = sample_question_pairs(cur_question_pool_dict)
                    sampled_question_pairs.update(sampled_cur_level_question_pairs)

    return sampled_question_pairs

def move_tensors_to_gpu(data):
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to('cuda')
    return data

def compose_questions_get_response(sampled_question_pairs, taxonomy_type, model, tokenizer, model_name='flan-t5-3b', variant_id=0):
    system_message = "Always answer with brief answers Yes, No, or I don't know. "

    print('current taxonomy type:', taxonomy_type, 'current variant id', variant_id)
    for sampled_question_key in sampled_question_pairs.keys():
        cur_question_cat = sampled_question_key
        total_num_questions = len(sampled_question_pairs[sampled_question_key])
        
        if variant_id == 0:
            folder_path = 'TaxoGlimpse/results/zero-shot/'+taxonomy_type+'/'+model_name+'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            cur_path = 'TaxoGlimpse/results/zero-shot/'+taxonomy_type+'/'+model_name+'/'+cur_question_cat+'.csv'
            #cur_path = 'TaxoGlimpse/results/zero-shot/'+taxonomy_type+'/'+model_name+'/'+cur_question_cat+'_updated.csv'

        else:
            folder_path = 'TaxoGlimpse/results/zero-shot-'+str(variant_id)+'/'+taxonomy_type+'/'+model_name+'/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            cur_path = 'TaxoGlimpse/results/zero-shot-'+str(variant_id)+'/'+taxonomy_type+'/'+model_name+'/'+cur_question_cat+'.csv'
        yes_total = 0
        no_total = 0
        dont_total = 0
        with open(cur_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            for (roottype, subtype) in tqdm(sampled_question_pairs[sampled_question_key]):
                dialog = system_message + compose_question_templates(taxonomy_type, subtype, roottype, variant_id) + '\nAnswer:'
                inputs = tokenizer(dialog, return_tensors="pt")
                inputs = move_tensors_to_gpu(inputs)
                outputs = model.generate(**inputs, max_new_tokens=20)
                flan_current_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip().lower().split('answer:')[-1]
                
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
        print('summary of exp:', cur_question_cat, 'total number of questions:', total_num_questions, ', yes total:', yes_total, ', no total:', no_total, ', miss total:', dont_total, 'acc', acc, 'missing rate', miss_rate)
        print('++++++++++++++++++++++++++++++++++++++++')
    

if __name__ == "__main__":
    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").cuda()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    #model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto")
    #tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
  
    question_pool_names = ['academic-acm', 'biology-NCBI', 'language-glottolog', 'medical-icd', 'shopping-amazon', 'shopping-google', 'shopping-ebay', 'general-schema', 'geography-geonames', 'medical-oae']
    
    
    variant_ids = [0,1,2]
    for question_pool_name in question_pool_names:
        for variant_id in variant_ids:
            cur_question_pairs = []
            sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy'], toroot_question_types=None, question_pool_name = question_pool_name)
            cur_question_pairs.append(sampled_question_pairs)
            for cur_question_pair in cur_question_pairs:
                compose_questions_get_response(cur_question_pair, taxonomy_type=question_pool_name, model=model, tokenizer=tokenizer, model_name='mistral', variant_id=variant_id)
