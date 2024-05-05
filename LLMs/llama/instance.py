import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
            if question_pool_name == 'shopping-amazon':
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
            elif question_pool_name == 'biology-NCBI':
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
            return sampled_question_pairs

    

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


def compose_questions_get_response_ctd(sampled_question_pairs, taxonomy_type, generator, ckpt_dir = 'llama-2-7b-chat/', exp_type='zero-shot', variant_id=0):
    #if exp_type == 'zero-shot' or exp_type == 'instance-zero' or exp_type == 'instance-full-zero':
    if exp_type == 'zero-shot' or exp_type == 'instance-zero-revision':
        system_message = "Always answer with brief answers Yes, No, or I don't know."
        dialog = {}
        dialog['system'] = system_message
        print('current taxonomy type:', taxonomy_type, 'current variant id', variant_id)
        for sampled_question_key in sampled_question_pairs.keys():
            cur_question_cat = sampled_question_key
            total_num_questions = len(sampled_question_pairs[sampled_question_key])
            if not variant_id == 0:
                cur_path = 'TaxoGlimpse/results/'+exp_type+'-'+str(variant_id)+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'.csv'
                folder_path = 'TaxoGlimpse/results/'+exp_type+'-'+str(variant_id)+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                #cur_path = 'TaxoGlimpse/results/'+exp_type+'-'+str(variant_id)+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'_updated.csv'
            else:
                cur_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'+cur_question_cat+'.csv'
                folder_path = 'TaxoGlimpse/results/'+exp_type+'/'+taxonomy_type+'/'+ckpt_dir.split('/')[0]+'/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
            yes_total = 0
            no_total = 0
            dont_total = 0
            with open(cur_path, 'a', newline='') as file:
                csv_writer = csv.writer(file)
                for (roottype, subtype) in tqdm(sampled_question_pairs[sampled_question_key]):

                    dialog['user'] = compose_question_templates(taxonomy_type, subtype, roottype, variant_id)

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
    

if __name__ == "__main__":
    ### choices are here, if 'level', you should pass types from level_question_types; else, pass types from toroot_question_types
    
    cur_model_dir = 'llama-2-13b-chat/'
    generator = Llama.build(
        ckpt_dir=cur_model_dir,
        tokenizer_path='tokenizer.model',
        max_seq_len=1024,
        max_batch_size=8)
    
    
    question_pool_names = ['biology-NCBI', 'language-glottolog', 'medical-icd', 'shopping-amazon', 'shopping-google', 'medical-oae']
    
    exp_type = 'instance-zero-revision'
    
    variant_ids = [0]
    if exp_type == 'instance-zero-revision':  
        for variant_id in variant_ids:
            for question_pool_name in question_pool_names:        
                cur_question_pairs = []
                sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive','negative_hard'], toroot_question_types=None, question_pool_name = question_pool_name, exp_type=exp_type)
                
                cur_question_pairs.append(sampled_question_pairs)
                for cur_question_pair in cur_question_pairs:
                    compose_questions_get_response_ctd(cur_question_pair, taxonomy_type=question_pool_name, generator=generator, ckpt_dir = cur_model_dir, exp_type=exp_type, variant_id=variant_id) # GPU29
    
        
    ### current file path: TaxoGlimpse/LLMs/llama/instance.py
    ### command: torchrun --nproc_per_node 1 instance.py 
    ### command: torchrun --nproc_per_node 2 instance.py
    ### command: torchrun --nproc_per_node 8 instance.py
