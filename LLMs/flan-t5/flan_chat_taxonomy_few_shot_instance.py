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
        sample_type = sample_key[15:]
        
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

def get_sampled_pairs(sub_question_type='level', level_question_types=['positive'], toroot_question_types=None, question_pool_name = 'academic-acm'):
    question_pools_base = 'TaxoGlimpse/question_pools/'
    cur_question_pool = question_pools_base+question_pool_name+'/'

    cur_question_pool_path = cur_question_pool+sub_question_type+'/'
    question_bank = {}
    sampled_question_pairs_list = []
    
    if sub_question_type == 'level':
        sampled_question_pairs = {}
        for level_question_type in level_question_types:
            cur_question_pool_path = cur_question_pool+'instance_typing/'
            cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
            cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
            question_bank.update(cur_question_pool_dict)
            cur_sampled = sample_question_pairs(cur_question_pool_dict)
            sampled_question_pairs.update(cur_sampled)
            sampled_question_pairs_list.append(cur_sampled)

    sampled_few_shots = sample_few_shot_examples(sampled_question_pairs_list, question_bank, n=5)
    return sampled_question_pairs, sampled_few_shots
### copied from evaluate_llama_taxonomy.py

def move_tensors_to_gpu(data):
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to('cuda')
    return data

def compose_questions_get_response(sampled_question_pairs, taxonomy_type, model, tokenizer, model_name='flan-t5-3b'):
    system_message = "Always answer with brief answers Yes, No, or I don't know. "
    print('current taxonomy type:', taxonomy_type)
    cur_sampled_question_pairs, sampled_few_shot = sampled_question_pairs
    for sampled_question_key in cur_sampled_question_pairs.keys():
        cur_question_cat = sampled_question_key
        total_num_questions = len(cur_sampled_question_pairs[sampled_question_key])
        cur_path = 'TaxoGlimpse/results/instance-few/'+taxonomy_type+'/'+model_name+'/'+cur_question_cat+'.csv'
        yes_total = 0
        no_total = 0
        dont_total = 0
        with open(cur_path, 'a', newline='') as file:
            csv_writer = csv.writer(file)
            for idx, (roottype_o, subtype_o) in enumerate(tqdm(cur_sampled_question_pairs[sampled_question_key])):
                cur_examples = sampled_few_shot[sampled_question_key][idx]
                cur_num_positive = cur_examples[-1]
                cur_positive_examples = cur_examples[:cur_num_positive]
                cur_negative_examples = cur_examples[cur_num_positive:-1]
                if taxonomy_type == 'academic-acm':
                    dialog = system_message
                    messages = []
                    for roottype, subtype in cur_positive_examples:
                        messages.append(('Example: Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)','Yes.'))
                    for roottype, subtype in cur_negative_examples:
                        messages.append(('Example: Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)','No.'))
                    random.shuffle(messages)
                    for message in messages:
                        dialog += message[0]
                        dialog += message[1]
                    dialog = dialog+'Is ' + subtype_o + ' computer science research concept a type of '+roottype_o +' computer science research concept? answer with (Yes/No/I don\'t know)' + '\nAnswer:'
                elif taxonomy_type == 'biology-NCBI':
                    dialog = system_message
                    messages = []
                    for roottype, subtype in cur_positive_examples:
                        messages.append(('Example: Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)','Yes.'))
                    for roottype, subtype in cur_negative_examples:
                        messages.append(('Example: Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)','No.'))
                    random.shuffle(messages)
                    for message in messages:
                        dialog += message[0]
                        dialog += message[1]
                    dialog = dialog+'Is ' + subtype_o + ' a type of '+roottype_o +'? answer with (Yes/No/I don\'t know)' + '\nAnswer:'
                elif taxonomy_type == 'language-glottolog':
                    dialog = system_message
                    messages = []
                    for roottype, subtype in cur_positive_examples:
                        messages.append(('Example: Is ' + subtype + ' language a type of '+roottype +' language? answer with (Yes/No/I don\'t know)','Yes.'))
                    for roottype, subtype in cur_negative_examples:
                        messages.append(('Example: Is ' + subtype + ' language a type of '+roottype +' language? answer with (Yes/No/I don\'t know)','No.'))
                    random.shuffle(messages)
                    for message in messages:
                        dialog += message[0]
                        dialog += message[1]
                    dialog = dialog+'Is ' + subtype_o + ' language a type of '+roottype_o +' language? answer with (Yes/No/I don\'t know)' + '\nAnswer:'
                elif taxonomy_type == 'medical-icd':
                    dialog = system_message
                    messages = []
                    for roottype, subtype in cur_positive_examples:
                        messages.append(('Example: Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)','Yes.'))
                    for roottype, subtype in cur_negative_examples:
                        messages.append(('Example: Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)','No.'))
                    random.shuffle(messages)
                    for message in messages:
                        dialog += message[0]
                        dialog += message[1]
                    dialog = dialog+'Is ' + subtype_o.lower() + ' a type of '+roottype_o.lower() +'? answer with (Yes/No/I don\'t know)' + '\nAnswer:'
                
                elif taxonomy_type == 'shopping-amazon':
                    dialog = system_message
                    messages = []
                    for roottype, subtype in cur_positive_examples:
                        messages.append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','Yes.'))
                    for roottype, subtype in cur_negative_examples:
                        messages.append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','No.'))
                    random.shuffle(messages)
                    for message in messages:
                        dialog += message[0]
                        dialog += message[1]
                    dialog = dialog+'Are ' + subtype_o + ' products a type of '+roottype_o +' products? answer with (Yes/No/I don\'t know)' + '\nAnswer:'
                elif taxonomy_type == 'shopping-google':
                    dialog = system_message
                    messages = []
                    for roottype, subtype in cur_positive_examples:
                        messages.append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','Yes.'))
                    for roottype, subtype in cur_negative_examples:
                        messages.append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','No.'))
                    random.shuffle(messages)
                    for message in messages:
                        dialog += message[0]
                        dialog += message[1]
                    dialog = dialog+'Are ' + subtype_o + ' products a type of '+roottype_o +' products? answer with (Yes/No/I don\'t know)' + '\nAnswer:'
                inputs = tokenizer(dialog, return_tensors="pt")
                inputs = move_tensors_to_gpu(inputs)
                outputs = model.generate(**inputs)
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
                cur_row = (roottype_o, subtype_o, flan_current_answer, decision)
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
        sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy', 'positive_to_root', 'negative_to_root'], toroot_question_types=None, question_pool_name = question_pool_name)
        cur_question_pairs.append(sampled_question_pairs)
        for cur_question_pair in cur_question_pairs:
            compose_questions_get_response(cur_question_pair, taxonomy_type=question_pool_name, model=model, tokenizer=tokenizer, model_name='flan-t5-3b')

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl").cuda() # 11B
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    question_pool_names = ['shopping-amazon']

    for question_pool_name in question_pool_names:
        cur_question_pairs = []
        sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy', 'positive_to_root', 'negative_to_root'], toroot_question_types=None, question_pool_name = question_pool_name)
        cur_question_pairs.append(sampled_question_pairs)
        for cur_question_pair in cur_question_pairs:
            compose_questions_get_response(cur_question_pair, taxonomy_type=question_pool_name, model=model, tokenizer=tokenizer, model_name='flan-t5-11b')
