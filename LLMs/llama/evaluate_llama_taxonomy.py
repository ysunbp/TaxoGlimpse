import os
import csv
import random
from typing import List, Optional
from llama import Llama, Dialog
from tqdm import tqdm
import torch.distributed as dist

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

def sample_few_shot_examples(cur_question_pool_dicts, question_bank, exp_type, n=5):
    ### cur_question_pool_dict: {key: level_{level}_question_type; value: [(parent, child),(parent, child),...]}
    out_example_dict = {}

    for cur_question_pool_dict in cur_question_pool_dicts:
        sample_key = list(cur_question_pool_dict.keys())[0]
        if exp_type == 'few-shot':
            sample_type = sample_key[8:]
        elif 'full' in exp_type:
            sample_type = sample_key[16:]
        else:
            sample_type = sample_key[15:]
        for cur_key in cur_question_pool_dict.keys():
            out_example_dict[cur_key] = []
            cur_key_level = cur_key.split('_')[1]
            cur_level_bank_positive = []
            cur_level_bank_negative = []
            out_example_dict[cur_key] = []
            if exp_type == 'few-shot':
                if sample_type == 'positive':
                    cur_level_bank_positive = question_bank['level_'+cur_key_level+'_'+sample_type]
                    cur_level_bank_negative = question_bank['level_'+cur_key_level+'_negative_easy']
                elif (sample_type == 'negative_easy') or (sample_type == 'negative_hard'):
                    cur_level_bank_positive = question_bank['level_'+cur_key_level+'_positive']
                    cur_level_bank_negative = question_bank['level_'+cur_key_level+'_'+sample_type]
                elif (sample_type == 'positive_to_root') or (sample_type == 'negative_to_root'):
                    cur_level_bank_positive = question_bank['level_'+cur_key_level+'_positive_to_root']
                    cur_level_bank_negative = question_bank['level_'+cur_key_level+'_negative_to_root']
            elif 'full' in exp_type:
                if sample_type == 'positive':
                    cur_level_bank_positive = question_bank['level_'+cur_key_level+'_'+sample_type]
                    cur_level_bank_negative = question_bank['level_'+cur_key_level+'_negative_easy']
                elif (sample_type == 'negative_easy') or (sample_type == 'negative_hard'):
                    cur_level_bank_positive = question_bank['level_'+cur_key_level+'_positive']
                    cur_level_bank_negative = question_bank['level_'+cur_key_level+'_'+sample_type]
            else:
                cur_key_level = 'instance'
                #print(sample_type)
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
    if exp_type == 'zero-shot' or exp_type == 'COT':
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
    
    elif ('instance' in exp_type) and (not 'few' in exp_type) and (not 'full' in exp_type):
        sampled_question_pairs = {}
        for level_question_type in level_question_types:
            cur_question_pool_path = cur_question_pool+'instance_typing/'
            cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
            cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
            sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
        return sampled_question_pairs

    elif exp_type == 'instance-few':
        question_bank = {}
        sampled_question_pairs_list = []
        sampled_question_pairs = {}
        for level_question_type in level_question_types:
            cur_question_pool_path = cur_question_pool+'instance_typing/'
            cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
            cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
            question_bank.update(cur_question_pool_dict)
            cur_sampled = sample_question_pairs(cur_question_pool_dict)
            sampled_question_pairs.update(cur_sampled)
            sampled_question_pairs_list.append(cur_sampled)
        sampled_few_shots = sample_few_shot_examples(sampled_question_pairs_list, question_bank, exp_type=exp_type, n=5)
        
        return sampled_question_pairs, sampled_few_shots
    
    elif ('instance-full' in exp_type):
        if exp_type == 'instance-full-zero' or exp_type == 'instance-full-COT':
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                cur_question_pool_path = cur_question_pool+'instance_full/'
                cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
                cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
            return sampled_question_pairs
        else:
            question_bank = {}
            sampled_question_pairs_list = []
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                cur_question_pool_path = cur_question_pool+'instance_full/'
                cur_csv_file = cur_question_pool_path + 'question_pool_full_' + level_question_type + '.csv'
                cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                question_bank.update(cur_question_pool_dict)
                cur_sampled = sample_question_pairs(cur_question_pool_dict)
                sampled_question_pairs.update(cur_sampled)
                sampled_question_pairs_list.append(cur_sampled)
            sampled_few_shots = sample_few_shot_examples(sampled_question_pairs_list, question_bank, exp_type=exp_type, n=5)
            return sampled_question_pairs, sampled_few_shots

    else: # few-shot
        question_bank = {}
        sampled_question_pairs_list = []
        if not question_pool_name == 'biology-NCBI':
            if sub_question_type == 'level':
                sampled_question_pairs = {}
                for level_question_type in level_question_types:
                    cur_csv_file = cur_question_pool_path + 'level_question_pool_full_' + level_question_type + '.csv'
                    cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                    question_bank.update(cur_question_pool_dict)
                    cur_sampled = sample_question_pairs(cur_question_pool_dict)
                    sampled_question_pairs.update(cur_sampled)
                    sampled_question_pairs_list.append(cur_sampled)
                    
            else:
                sampled_question_pairs = {}
                for toroot_question_type in toroot_question_types:
                    cur_csv_file = cur_question_pool_path + 'question_pool_full_' + toroot_question_type + '.csv'
                    cur_question_pool_dict = load_csv_file(cur_csv_file, toroot_question_type)
                    question_bank.update(cur_question_pool_dict)
                    cur_sampled = sample_question_pairs(cur_question_pool_dict)
                    sampled_question_pairs.update(cur_sampled)
                    sampled_question_pairs_list.append(cur_sampled)
        else:
            if sub_question_type == 'level':
                sampled_question_pairs = {}
                for level_question_type in level_question_types:
                    cur_sampled = {}
                    for cur_level in range(1, question_pool_levels['ncbi']+1):
                        cur_csv_file = cur_question_pool_path + 'level_question_pool_full_level_' + str(cur_level) + '_' + level_question_type + '.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                        question_bank.update(cur_question_pool_dict)
                        sampled_cur_level_question_pairs = sample_question_pairs(cur_question_pool_dict)
                        cur_sampled.update(sampled_cur_level_question_pairs)
                        sampled_question_pairs.update(sampled_cur_level_question_pairs)
                    sampled_question_pairs_list.append(cur_sampled)
                    
            else:
                sampled_question_pairs = {}
                for toroot_question_type in toroot_question_types:
                    cur_sampled = {}
                    for cur_level in range(1, question_pool_levels['ncbi']+1):
                        cur_csv_file = cur_question_pool_path + 'question_pool_full_level_' + str(cur_level) + '_' + toroot_question_type + '.csv'
                        cur_question_pool_dict = load_csv_file(cur_csv_file, toroot_question_type)
                        question_bank.update(cur_question_pool_dict)
                        sampled_cur_level_question_pairs = sample_question_pairs(cur_question_pool_dict)
                        cur_sampled.update(sampled_cur_level_question_pairs)
                        sampled_question_pairs.update(sampled_cur_level_question_pairs)
                    sampled_question_pairs_list.append(cur_sampled)

        sampled_few_shots = sample_few_shot_examples(sampled_question_pairs_list, question_bank, exp_type=exp_type, n=5)
        
        return sampled_question_pairs, sampled_few_shots

def llama_answer(
    dialog, generator,
    temperature = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    

    dialogs: List[Dialog] = [
        [ {"role": "system", "content": dialog['system']},
            {"role": "user", "content": dialog['user']}]]  
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    output = []
    for dialog, result in zip(dialogs, results):
        output.append(result['generation']['content'])
    return output
    
def llama_answer_few_shot(
    dialog, generator,
    temperature = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    
    dialog_content = [ {"role": "system", "content": dialog['system']}]

    random.shuffle(dialog['message'])
    for message in dialog['message']:
        dialog_content.append({"role": "user", "content": message[0]})
        dialog_content.append({"role": "assistant", "content": message[1]})
    dialog_content.append({"role": "user", "content": dialog['user']})
    dialogs = [dialog_content]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    output = []
    for dialog, result in zip(dialogs, results):
        output.append(result['generation']['content'])
    return output

def llama_answer_COT(
    dialog, generator,
    temperature = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    

    dialogs: List[Dialog] = [
        [{"role": "system", "content": dialog['system']}, {"role": "user", "content": dialog['user']+"Let's think step by step."}]]  
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    output = []
    for dialog, result in zip(dialogs, results):
        output.append(result['generation']['content'])
    return output

def compose_questions_get_response_ctd(sampled_question_pairs, taxonomy_type, generator, ckpt_dir = 'llama-2-7b-chat/', exp_type='zero-shot'):
    if exp_type == 'zero-shot' or exp_type == 'instance-zero' or exp_type == 'instance-full-zero':
        system_message = "Always answer with brief answers Yes, No, or I don't know."
        dialog = {}
        dialog['system'] = system_message
        print('current taxonomy type:', taxonomy_type)
        for sampled_question_key in sampled_question_pairs.keys():
            cur_question_cat = sampled_question_key
            total_num_questions = len(sampled_question_pairs[sampled_question_key])
            cur_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'.csv'
            yes_total = 0
            no_total = 0
            dont_total = 0
            with open(cur_path, 'a', newline='') as file:
                csv_writer = csv.writer(file)
                for (roottype, subtype) in tqdm(sampled_question_pairs[sampled_question_key]):
                    if taxonomy_type == 'academic-acm':
                        dialog['user'] = 'Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'biology-NCBI':
                        dialog['user'] = 'Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'language-glottolog':
                        dialog['user'] = 'Is ' + subtype + ' language a type of '+roottype +' language? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'medical-icd':
                        dialog['user'] = 'Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'shopping-amazon':
                        dialog['user'] = 'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'shopping-google':
                        dialog['user'] = 'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)'
                    llama_current_answer = llama_answer(dialog, generator, temperature = 0)[0].strip().lower()

                    if 'yes' in llama_current_answer:
                        decision = 1
                        yes_total += 1
                    elif 'know' in llama_current_answer:
                        decision = 2
                        dont_total += 1
                    else:
                        decision = 0
                        no_total += 1
                    cur_row = (roottype, subtype, llama_current_answer, decision)
                    if dist.get_rank() == 0:
                        csv_writer.writerow(cur_row)

            if 'positive' in cur_question_cat:
                acc = yes_total/total_num_questions
            else:
                acc = no_total/total_num_questions
            miss_rate = dont_total/total_num_questions
            print('summary of exp:', cur_question_cat, ', total number of questions:', total_num_questions, ', yes total:', yes_total, ', no total:', no_total, ', miss total:', dont_total, 'acc', acc, 'missing rate', miss_rate)
            print('++++++++++++++++++++++++++++++++++++++++')
    elif exp_type == 'few-shot' or exp_type == 'instance-few' or exp_type == 'instance-full-few': #TODO: modify the question composing part
        cur_sampled_question_pairs, sampled_few_shot = sampled_question_pairs
        system_message = "Always answer with brief answers Yes, No, or I don't know."
        dialog = {}
        dialog['system'] = system_message
        print('current taxonomy type:', taxonomy_type)
        for sampled_question_key in cur_sampled_question_pairs.keys():
            cur_question_cat = sampled_question_key
            total_num_questions = len(cur_sampled_question_pairs[sampled_question_key])
            cur_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'.csv'
            yes_total = 0
            no_total = 0
            dont_total = 0
            with open(cur_path, 'a', newline='') as file:
                csv_writer = csv.writer(file)
                for idx, (roottype_o, subtype_o) in enumerate(tqdm(cur_sampled_question_pairs[sampled_question_key])):
                    dialog['message'] = []
                    cur_examples = sampled_few_shot[sampled_question_key][idx]
                    cur_num_positive = cur_examples[-1]
                    cur_positive_examples = cur_examples[:cur_num_positive]
                    cur_negative_examples = cur_examples[cur_num_positive:-1]
                    if taxonomy_type == 'academic-acm':
                        for roottype, subtype in cur_positive_examples:
                            dialog['message'].append(('Example: Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)','Yes.'))
                        for roottype, subtype in cur_negative_examples:
                            dialog['message'].append(('Example: Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)','No.'))
                        dialog['user'] = 'Is ' + subtype_o + ' computer science research concept a type of '+roottype_o +' computer science research concept? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'biology-NCBI':
                        for roottype, subtype in cur_positive_examples:
                            dialog['message'].append(('Example: Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)','Yes.'))
                        for roottype, subtype in cur_negative_examples:
                            dialog['message'].append(('Example: Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)','No.'))
                        dialog['user'] = 'Is ' + subtype_o + ' a type of '+roottype_o +'? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'language-glottolog':
                        for roottype, subtype in cur_positive_examples:
                            dialog['message'].append(('Example: Is ' + subtype + ' language a type of '+roottype +' language? answer with (Yes/No/I don\'t know)','Yes.'))
                        for roottype, subtype in cur_negative_examples:
                            dialog['message'].append(('Example: Is ' + subtype + ' language a type of '+roottype +' language? answer with (Yes/No/I don\'t know)','No.'))
                        dialog['user'] = 'Is ' + subtype_o + ' language a type of '+roottype_o +' language? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'medical-icd':
                        for roottype, subtype in cur_positive_examples:
                            dialog['message'].append(('Example: Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)','Yes.'))
                        for roottype, subtype in cur_negative_examples:
                            dialog['message'].append(('Example: Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)','No.'))
                        dialog['user'] = 'Is ' + subtype_o.lower() + ' a type of '+roottype_o.lower() +'? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'shopping-amazon':
                        for roottype, subtype in cur_positive_examples:
                            dialog['message'].append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','Yes.'))
                        for roottype, subtype in cur_negative_examples:
                            dialog['message'].append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','No.'))
                        dialog['user'] = 'Are ' + subtype_o + ' products a type of '+roottype_o +' products? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'shopping-google':
                        for roottype, subtype in cur_positive_examples:
                            dialog['message'].append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','Yes.'))
                        for roottype, subtype in cur_negative_examples:
                            dialog['message'].append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','No.'))
                        dialog['user'] = 'Are ' + subtype_o + ' products a type of '+roottype_o +' products? answer with (Yes/No/I don\'t know)'
                    llama_current_answer = llama_answer_few_shot(dialog, generator, temperature = 0)[0].strip().lower()

                    if 'yes' in llama_current_answer:
                        decision = 1
                        yes_total += 1
                    elif 'know' in llama_current_answer:
                        decision = 2
                        dont_total += 1
                    else:
                        decision = 0
                        no_total += 1
                    cur_row = (roottype_o, subtype_o, llama_current_answer, decision)
                    if dist.get_rank() == 0:
                        csv_writer.writerow(cur_row)

            if 'positive' in cur_question_cat:
                acc = yes_total/total_num_questions
            else:
                acc = no_total/total_num_questions
            miss_rate = dont_total/total_num_questions
            print('summary of exp:', cur_question_cat, ', total number of questions:', total_num_questions, ', yes total:', yes_total, ', no total:', no_total, ', miss total:', dont_total, 'acc', acc, 'missing rate', miss_rate)
            print('++++++++++++++++++++++++++++++++++++++++')
    elif exp_type == 'COT' or exp_type == 'instance-COT' or exp_type == 'instance-full-COT':
        system_message = "Always answer with Yes, No, or I don't know."
        dialog = {}
        dialog['system'] = system_message
        print('current taxonomy type:', taxonomy_type)
        for sampled_question_key in sampled_question_pairs.keys():
            cur_question_cat = sampled_question_key
            total_num_questions = len(sampled_question_pairs[sampled_question_key])
            cur_path = '../results/'+exp_type+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'.csv'
            yes_total = 0
            no_total = 0
            dont_total = 0
            with open(cur_path, 'a', newline='') as file:
                csv_writer = csv.writer(file)
                for (roottype, subtype) in tqdm(sampled_question_pairs[sampled_question_key]):
                    if taxonomy_type == 'academic-acm':
                        dialog['user'] = 'Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'biology-NCBI':
                        dialog['user'] = 'Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'language-glottolog':
                        dialog['user'] = 'Is ' + subtype + ' language a type of '+roottype +' language? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'medical-icd':
                        dialog['user'] = 'Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'shopping-amazon':
                        dialog['user'] = 'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)'
                    elif taxonomy_type == 'shopping-google':
                        dialog['user'] = 'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)'
                    llama_current_answer = llama_answer_COT(dialog, generator, temperature = 0)[0].strip().lower()

                    if 'yes' in llama_current_answer:
                        decision = 1
                        yes_total += 1
                    elif 'know' in llama_current_answer:
                        decision = 2
                        dont_total += 1
                    else:
                        decision = 0
                        no_total += 1
                    cur_row = (roottype, subtype, llama_current_answer, decision)
                    if dist.get_rank() == 0:
                        csv_writer.writerow(cur_row)

            if 'positive' in cur_question_cat:
                acc = yes_total/total_num_questions
            else:
                acc = no_total/total_num_questions
            miss_rate = dont_total/total_num_questions
            print('summary of exp:', cur_question_cat, ', total number of questions:', total_num_questions, ', yes total:', yes_total, ', no total:', no_total, ', miss total:', dont_total, 'acc', acc, 'missing rate', miss_rate)
            print('++++++++++++++++++++++++++++++++++++++++')
    

if __name__ == "__main__":
    cur_model_dir = 'llama-2-7b-chat/'
    generator = Llama.build(
        ckpt_dir=cur_model_dir,
        tokenizer_path='tokenizer.model',
        max_seq_len=1024,
        max_batch_size=8)
    ########################## main experiment ##########################
    exp_types = ['zero-shot', 'few-shot', 'COT']
    for exp_type in exp_types:
        question_pool_names = ['academic-acm','biology-NCBI','language-glottolog', 'medical-icd', 'shopping-amazon', 'shopping-google']  
        
        if exp_type == 'zero-shot' or exp_type == 'COT':  
            for question_pool_name in question_pool_names:
                cur_question_pairs = []
                sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy'], toroot_question_types=None, question_pool_name = question_pool_name)
                cur_question_pairs.append(sampled_question_pairs)
                #sampled_question_pairs = get_sampled_pairs(sub_question_type='toroot', level_question_types=None, toroot_question_types=['positive_to_root', 'negative_to_root'], question_pool_name = question_pool_name)
                #cur_question_pairs.append(sampled_question_pairs)
                for cur_question_pair in cur_question_pairs:
                    compose_questions_get_response_ctd(cur_question_pair, taxonomy_type=question_pool_name, generator=generator, ckpt_dir = cur_model_dir, exp_type=exp_type) # GPU29
        else:
            for question_pool_name in question_pool_names:
                cur_question_pairs = []
                sampled_question_pairs, sampled_few_shot = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy'], toroot_question_types=None, question_pool_name = question_pool_name, exp_type=exp_type)
                cur_question_pairs.append((sampled_question_pairs, sampled_few_shot))
                #sampled_question_pairs, sampled_few_shot = get_sampled_pairs(sub_question_type='toroot', level_question_types=None, toroot_question_types=['positive_to_root', 'negative_to_root'], question_pool_name = question_pool_name, exp_type=exp_type)
                #cur_question_pairs.append((sampled_question_pairs, sampled_few_shot))
                for cur_question_pair, cur_few_shot in cur_question_pairs:
                    compose_questions_get_response_ctd((cur_question_pair, cur_few_shot), taxonomy_type=question_pool_name, generator=generator, ckpt_dir = cur_model_dir, exp_type=exp_type) # GPU29

    ########################## instance typing ##########################
    exp_types = ['instance-full-zero', 'instance-full-few', 'instance-full-COT', 'instance-zero', 'instance-few', 'instance-COT']
    question_pool_names = ['academic-acm','biology-NCBI','language-glottolog', 'medical-icd', 'shopping-amazon', 'shopping-google']  
    
    for exp_type in exp_types:
        print('current exp type', exp_type)
        for question_pool_name in question_pool_names:
            # evaluate the last layer to root and the last layer to the root parent
            
            if exp_type == 'instance-few':
            cur_question_pairs = []
            sampled_question_pairs, sampled_few_shot = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy', 'positive_to_root', 'negative_to_root'], toroot_question_types=None, question_pool_name = question_pool_name, exp_type=exp_type)
            cur_question_pairs.append((sampled_question_pairs, sampled_few_shot))
            for cur_question_pair, cur_few_shot in cur_question_pairs:
                compose_questions_get_response_ctd((cur_question_pair, cur_few_shot), taxonomy_type=question_pool_name, generator=generator, ckpt_dir = cur_model_dir, exp_type=exp_type) # GPU29
            elif exp_type == 'instance-zero' or exp_type == 'instance-COT':
                sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy', 'positive_to_root', 'negative_to_root'], toroot_question_types=None, question_pool_name = question_pool_name, exp_type=exp_type)
                compose_questions_get_response_ctd(sampled_question_pairs, taxonomy_type=question_pool_name, generator=generator, ckpt_dir = cur_model_dir, exp_type=exp_type) # GPU29 

            # evaluate the last layer to all the parents except the root
            elif exp_type == 'instance-full-zero' or exp_type == 'instance-full-COT':
                sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy'], toroot_question_types=None, question_pool_name = question_pool_name, exp_type=exp_type)                
                compose_questions_get_response_ctd(sampled_question_pairs, taxonomy_type=question_pool_name, generator=generator, ckpt_dir = cur_model_dir, exp_type=exp_type) # GPU29
            elif exp_type == 'instance-full-few':
                cur_question_pairs = []
                sampled_question_pairs, sampled_few_shot = get_sampled_pairs(sub_question_type='level', level_question_types=['positive','negative_hard', 'negative_easy'], toroot_question_types=None, question_pool_name = question_pool_name, exp_type=exp_type)                
                cur_question_pairs.append((sampled_question_pairs, sampled_few_shot))
                for cur_question_pair, cur_few_shot in cur_question_pairs:
                    compose_questions_get_response_ctd((cur_question_pair, cur_few_shot), taxonomy_type=question_pool_name, generator=generator, ckpt_dir = cur_model_dir, exp_type=exp_type) # GPU29
        
        ### command: torchrun --nproc_per_node 1 evaluate_llama_taxonomy.py >> ../logs/llama-2-7b-chat/level.txt
        ### command: torchrun --nproc_per_node 2 evaluate_llama_taxonomy.py >> ../logs/llama-2-13b-chat/level.txt
        ### command: torchrun --nproc_per_node 8 evaluate_llama_taxonomy.py >> ../logs/llama-2-70b-chat/level.txt