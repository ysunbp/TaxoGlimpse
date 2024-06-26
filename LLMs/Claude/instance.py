import csv
import random
from tqdm import tqdm
from litellm import completion
import os
# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "xxxxxx"
import time
import os

def generateResponse(prompt, gpt_name, exp_type):
    
    '''
    Input:
        prompt (str): composed prompt.
    Output:
        response (str): response from GPT-4.

    This function takes composed prompt as input and returns the response from GPT.
    '''
    if exp_type == 'instance-zero-revision':
        messages = [{"role": "system", "content": prompt['system']}, {"role": "user", "content": prompt['user']}]

    response = completion(model="claude-3-opus-20240229", messages=messages, api_base="https://api.openai-proxy.org/anthropic/v1/messages", temperature=0)
    return response['choices'][0]['message']['content']


def compute_cur_level_size(total_samples):
    # 95% confidence, margin of error 5%
    return int(((1.96**2*0.5*(1-0.5)/0.05**2)/(1+(1.96**2*0.5*(1-0.5)/(0.05**2*total_samples))))+1)

def setup_seed(seed):
    random.seed(seed)

def load_csv_file_medical_toroot(csv_path, question_type):
    question_pools = {}
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for idx, row in enumerate(csvreader):
            if not row[3] == '3':
                continue
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

def check_dup(sampled_list, item):
    flag = False
    for sampled_item in sampled_list:
        if sampled_item[1] == item[1]:
            flag = True
    return flag

def sample_few_shot_examples(cur_question_pool_dicts, question_bank, n=5):
    ### cur_question_pool_dict: {key: level_{level}_question_type; value: [(parent, child),(parent, child),...]}
    out_example_dict = {}

    for cur_question_pool_dict in cur_question_pool_dicts:
        sample_key = list(cur_question_pool_dict.keys())[0]
        sample_type = sample_key[8:]
        
        for cur_key in cur_question_pool_dict.keys():
            out_example_dict[cur_key] = []
            cur_key_level = cur_key.split('_')[1]
            cur_level_bank_positive = []
            cur_level_bank_negative = []
            out_example_dict[cur_key] = []
            if sample_type == 'positive':
                cur_level_bank_positive = question_bank['level_'+cur_key_level+'_'+sample_type]
                cur_level_bank_negative = question_bank['level_'+cur_key_level+'_negative_easy']
            elif (sample_type == 'negative_easy') or (sample_type == 'negative_hard'):
                cur_level_bank_positive = question_bank['level_'+cur_key_level+'_positive']
                cur_level_bank_negative = question_bank['level_'+cur_key_level+'_'+sample_type]
            elif (sample_type == 'positive_to_root') or (sample_type == 'negative_to_root'):
                cur_level_bank_positive = question_bank['level_'+cur_key_level+'_positive_to_root']
                cur_level_bank_negative = question_bank['level_'+cur_key_level+'_negative_to_root']
            for i in range(len(cur_question_pool_dict[cur_key])):
                setup_seed(i)
                num_positive = random.randrange(0,n+1)
                setup_seed(i)
                cur_samples = random.sample(cur_level_bank_positive, num_positive)
                setup_seed(i)
                cur_samples += random.sample(cur_level_bank_negative, n-num_positive)
                j = i
                while check_dup(cur_samples, cur_question_pool_dict[cur_key][i]):
                    j += 100
                    setup_seed(j)
                    cur_samples = random.sample(cur_level_bank_positive, num_positive)
                    setup_seed(j)
                    cur_samples += random.sample(cur_level_bank_negative, n-num_positive)
                cur_samples.append(num_positive)
                out_example_dict[cur_key].append(cur_samples)
    #print(out_example_dict)
    return out_example_dict

def get_sampled_pairs(sub_question_type='level', level_question_types=['positive'], toroot_question_types=None, question_pool_name = 'academic-acm', exp_type = 'zero-shot'):
    question_pool_levels = {'acm':4, 'ncbi':6, 'glottolog':5, 'icd':3, 'amazon':4, 'google':4}
    question_pools_base = 'TaxoGlimpse/question_pools/'
    cur_question_pool = question_pools_base+question_pool_name+'/'

    cur_question_pool_path = cur_question_pool+sub_question_type+'/'
    if exp_type == 'instance-zero-revision': # revision modified
        if not question_pool_name in ['biology-NCBI', 'medical-icd', 'shopping-amazon']:
            
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                cur_question_pool_path = cur_question_pool+'instance_full/'
                cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
                cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
            return sampled_question_pairs
        else:
            if question_pool_name == 'biology-NCBI':
                sampled_question_pairs = {}
                for level_question_type in level_question_types:
                    cur_question_pool_path = cur_question_pool+'instance_full/'
                    to_root_question_pool_path = cur_question_pool + 'toroot/'
                    if level_question_type == 'positive':
                        cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                        cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '_partial.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                        cur_csv_file = to_root_question_pool_path + 'question_pool_full_level_6_' + level_question_type + '_to_root.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                    elif level_question_type == 'negative_hard':
                        cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '_all.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                        cur_csv_file = to_root_question_pool_path + 'question_pool_full_level_6_negative_to_root.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                    
            elif question_pool_name == 'medical-icd':
                sampled_question_pairs = {}
                for level_question_type in level_question_types:
                    cur_question_pool_path = cur_question_pool+'instance_full/'
                    to_root_question_pool_path = cur_question_pool + 'toroot/'
                    if level_question_type == 'positive':
                        cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                        cur_csv_file = to_root_question_pool_path + 'question_pool_full_' + level_question_type + '_to_root.csv'
                        cur_question_pool_dict = load_csv_file_medical_toroot(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))

                    elif level_question_type == 'negative_hard':
                        cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                        cur_csv_file = to_root_question_pool_path + 'question_pool_full_negative_to_root.csv'
                        cur_question_pool_dict = load_csv_file_medical_toroot(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
            
            elif question_pool_name == 'shopping-amazon':
                sampled_question_pairs = {}
                for level_question_type in level_question_types:
                    cur_question_pool_path = cur_question_pool+'instance_full/'
                    to_root_question_pool_path = cur_question_pool + 'instance_typing/'
                    if level_question_type == 'positive':
                        cur_csv_file = cur_question_pool_path + 'question_pool_full_updated_' + level_question_type + '.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                        cur_csv_file = to_root_question_pool_path + 'question_pool_full_' + level_question_type + '_to_root.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                    elif level_question_type == 'negative_hard':
                        cur_csv_file = cur_question_pool_path + 'question_pool_full_updated_' + level_question_type + '.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
                        cur_csv_file = to_root_question_pool_path + 'question_pool_full_negative_to_root.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
            return sampled_question_pairs


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




def compose_questions_get_response_ctd(sampled_question_pairs, taxonomy_type, ckpt_dir, exp_type='zero-shot', variant_id=0):
    if exp_type == 'zero-shot' or 'instance-zero-revision':
        system_message = "Always answer with brief answers Yes, No, or I don't know."
        dialog = {}
        dialog['system'] = system_message
        print('current taxonomy type:', taxonomy_type, 'current variant id', variant_id)
        for sampled_question_key in sampled_question_pairs.keys():
            cur_question_cat = sampled_question_key
            total_num_questions = len(sampled_question_pairs[sampled_question_key])
            #cur_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'.csv'
            model_name = ckpt_dir.split('/')[0]
            if variant_id == 0:
                folder_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+model_name+'/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                cur_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+model_name+'/'+cur_question_cat+'.csv'
                #cur_path = 'TaxoGlimpse/results/zero-shot/'+taxonomy_type+'/'+model_name+'/'+cur_question_cat+'_updated.csv'

            else:
                folder_path = 'TaxoGlimpse/results/'+exp_type+'-'+str(variant_id)+'/'+taxonomy_type+'/'+model_name+'/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                cur_path = 'TaxoGlimpse/results/'+exp_type+'-'+str(variant_id)+'/'+taxonomy_type+'/'+model_name+'/'+cur_question_cat+'.csv'
            
            yes_total = 0
            no_total = 0
            dont_total = 0
            with open(cur_path, 'a', newline='') as file:
                csv_writer = csv.writer(file)
                for (roottype, subtype) in tqdm(sampled_question_pairs[sampled_question_key]):
                    dialog['user'] = compose_question_templates(taxonomy_type, subtype, roottype, variant_id)
                    retry = 0
                    while True:
                        try:
                            gpt_current_answer = generateResponse(dialog, ckpt_dir, exp_type).lower()
                            time.sleep(0.4)
                            break
                        except Exception as error:
                            time.sleep(0.4)
                            retry += 1
                            if retry > 3:
                                gpt_current_answer = 'i don\'t know'
                                break

                    if 'yes' in gpt_current_answer:
                        decision = 1
                        yes_total += 1
                    elif 'know' in gpt_current_answer:
                        decision = 2
                        dont_total += 1
                    else:
                        decision = 0
                        no_total += 1
                    cur_row = (roottype, subtype, gpt_current_answer, decision)
                    csv_writer.writerow(cur_row)
            if 'positive' in cur_question_cat:
                acc = yes_total/total_num_questions
            else:
                acc = no_total/total_num_questions
            miss_rate = dont_total/total_num_questions
            print('summary of exp:', cur_question_cat, ', total number of questions:', total_num_questions, ', yes total:', yes_total, ', no total:', no_total, ', miss total:', dont_total, 'acc', acc, 'missing rate', miss_rate)
            print('++++++++++++++++++++++++++++++++++++++++')
    
if __name__ == "__main__":
    GPT_name = "claude"
    
    question_pool_names = ['shopping-amazon','shopping-google','biology-NCBI', 'language-glottolog', 'medical-icd','medical-oae']  

    variant_ids = [0]
    exp_type = 'instance-zero-revision'
    
    if exp_type == 'instance-zero-revision':  
        for question_pool_name in question_pool_names:
            for variant_id in variant_ids:
                cur_question_pairs = []
                sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard'], toroot_question_types=None, question_pool_name = question_pool_name, exp_type=exp_type)
                cur_question_pairs.append(sampled_question_pairs)
                
                for cur_question_pair in cur_question_pairs:
                    compose_questions_get_response_ctd(cur_question_pair, taxonomy_type=question_pool_name, ckpt_dir = GPT_name, exp_type=exp_type, variant_id=variant_id) # GPU29
        
        
