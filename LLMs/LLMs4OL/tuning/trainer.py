import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
from configs import BaseConfig
import argparse
from datareader import DataReader
from dataset import DatasetFactory
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import random
import csv
from tqdm import tqdm


def preprocess_function(sample, padding="max_length"):
    inputs = [item for item in sample["text"]]

    # tokenize inputs
    model_inputs = TOKENIZER(inputs, max_length=CONFIG.max_source_length, padding=padding, truncation=True)

    # tokenize targets
    labels = TOKENIZER(text_target=sample["label"], max_length=CONFIG.max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the
    # labels by -100 when we want to ignore padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != TOKENIZER.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_cur_level_size(total_samples):
    # 95% confidence, margin of error 5%
    return int(((1.96**2*0.5*(1-0.5)/0.05**2)/(1+(1.96**2*0.5*(1-0.5)/(0.05**2*total_samples))))+1)

def setup_seed(seed):
    random.seed(seed)

def sample_question_pairs(question_pool_dict):
    sampled_question_pairs = {}
    remained = []
    total_sample_size = 0
    for question_pool_key in question_pool_dict.keys():
        total_sample_size += compute_cur_level_size(len(question_pool_dict[question_pool_key]))
    for question_pool_key in question_pool_dict.keys():
        sample_size = compute_cur_level_size(len(question_pool_dict[question_pool_key]))
        setup_seed(20)
        obtained = random.sample(question_pool_dict[question_pool_key], sample_size)
        remained += [item for item in question_pool_dict[question_pool_key] if item not in obtained]
    
    first_question_pool_key = list(question_pool_dict.keys())[0]
    if 'positive' in first_question_pool_key:
        key = 'positive'
    elif 'negative' in first_question_pool_key:
        key = 'negative'
    elif 'mcq' in first_question_pool_key and 'hard' in first_question_pool_key:
        key = 'mcq_hard'
    else:
        key = 'mcq_easy'
    setup_seed(20)        
    sampled_question_pairs[key] = random.sample(remained, min(int(total_sample_size*0.25), len(remained)))

    return sampled_question_pairs

def sample_question_pairs_bio(question_pool_dict):
    sampled_question_pairs = {}
    remained = []
    for question_pool_key in question_pool_dict.keys():
        sample_size = compute_cur_level_size(len(question_pool_dict[question_pool_key]))
        setup_seed(20)
        obtained = random.sample(question_pool_dict[question_pool_key], sample_size)
        remained += [item for item in question_pool_dict[question_pool_key] if item not in obtained]
        setup_seed(20)        
        sampled_question_pairs[question_pool_key] = random.sample(remained, min(int(0.25*sample_size), len(remained)))

    return sampled_question_pairs

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
        elif sub_question_type == 'mcq':
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                cur_csv_file = cur_question_pool_path + 'question_pool_full_mcq_'+level_question_type+'.csv'
                cur_question_pool_dict = load_csv_file_mcq(cur_csv_file, level_question_type)
                sampled_question_pairs.update(sample_question_pairs(cur_question_pool_dict))
    else:
        if sub_question_type == 'level':
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                for cur_level in range(1, question_pool_levels['ncbi']+1):
                    cur_csv_file = cur_question_pool_path + 'level_question_pool_full_level_' + str(cur_level) + '_' + level_question_type + '.csv'
                    cur_question_pool_dict = load_csv_file(cur_csv_file, level_question_type)
                    sampled_cur_level_question_pairs = sample_question_pairs_bio(cur_question_pool_dict)
                    sampled_question_pairs.update(sampled_cur_level_question_pairs)
        elif sub_question_type == 'mcq':
            sampled_question_pairs = {}
            for level_question_type in level_question_types:
                for cur_level in range(1, question_pool_levels['ncbi']+1):
                    cur_csv_file = cur_question_pool_path + 'question_pool_full_level_' + str(cur_level) + '_mcq_' + level_question_type + '.csv'
                    cur_question_pool_dict = load_csv_file_mcq(cur_csv_file, level_question_type)
                    sampled_cur_level_question_pairs = sample_question_pairs_bio(cur_question_pool_dict)
                    sampled_question_pairs.update(sampled_cur_level_question_pairs)

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

    return question, correct_index




def load_training_data(kb_name):
    source_text, target_text = [], []
    question_pool_name = kb_name
    sampled_question_pairs_tf = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard'], toroot_question_types=None, question_pool_name = question_pool_name)
    sampled_question_pairs_mcq = get_sampled_pairs(sub_question_type='mcq', level_question_types=['hard', 'easy'], toroot_question_types=None, question_pool_name = question_pool_name)

    system_message = "Always answer with brief answers Yes, No, or I don't know. "
    for sampled_question_key in sampled_question_pairs_tf.keys():
        #print(len(sampled_question_pairs_tf[sampled_question_key]))
        for (roottype, subtype) in tqdm(sampled_question_pairs_tf[sampled_question_key]):
            variant_ids = [0,1,2]
            for variant_id in variant_ids:
                dialog = system_message + compose_question_templates(kb_name, subtype, roottype, variant_id) + '\nAnswer:'
                source_text.append(dialog)
                if 'positive' in sampled_question_key:
                    target_text.append("Yes.")
                else:
                    target_text.append("No.")


    system_message = "Always answer with brief answers A), B), C), D), or I don't know."
    answer_notes = ['A)', 'B)', 'C)', 'D)']
    for sampled_question_key in sampled_question_pairs_mcq.keys():
        #print(len(sampled_question_pairs_mcq[sampled_question_key]))
        for (roottype, subtype, alternatives) in tqdm(sampled_question_pairs_mcq[sampled_question_key]):
            variant_ids = [0,1,2]
            for variant_id in variant_ids:
                question, ground_truth_idx = compose_question_templates_mcq(kb_name, subtype, roottype, alternatives, variant_id)
                dialog = system_message + question + '\nAnswer:'
                source_text.append(dialog)
                target_text.append(answer_notes[ground_truth_idx])

    return source_text, target_text


if __name__=="__main__":
    question_pool_names = ['academic-acm', 'biology-NCBI', 'language-glottolog', 'medical-icd', 'shopping-amazon', 'shopping-google', 'shopping-ebay', 'general-schema', 'geography-geonames', 'medical-oae']
    for question_pool_name in question_pool_names:
        source_text, target_text = load_training_data(question_pool_name)
        print('training set size',len(source_text))
        parser = argparse.ArgumentParser()
        parser.add_argument("--kb_name", default=question_pool_name)
        parser.add_argument("--model_to_train", default='flan_t5_xl')
        parser.add_argument("--num_train_epochs", type=int, default=15)

        args = parser.parse_args()
        print("args:", args)
        CONFIG = BaseConfig().get_args(kb_name=args.kb_name, model_to_train=args.model_to_train)

        # loading dataset
        #dataset_json = DataReader.load_json(path=CONFIG.fsl_train_data)
        #print(dataset_json)
        #source_text, target_text = DatasetFactory(dataset=args.kb_name).build_samples(dataset=dataset_json)
        #print('source', len(source_text), source_text) # indeed a list of questions
        #print('target', target_text) # a list of answers
        dataset = DatasetDict({'train': Dataset.from_dict({'label': target_text, 'text': source_text})})

        TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.model_input_path)
        tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text", "label"])
        MODEL = AutoModelForSeq2SeqLM.from_pretrained(CONFIG.model_input_path, device_map='auto')

        # we want to ignore tokenizer pad token in the loss
        data_collator = DataCollatorForSeq2Seq(
            TOKENIZER,
            model=MODEL,
            label_pad_token_id=CONFIG.label_pad_token_id,
            pad_to_multiple_of=8
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=CONFIG.output_log_dir,
            auto_find_batch_size=CONFIG.auto_find_batch_size,
            learning_rate=CONFIG.learning_rate,
            num_train_epochs=args.num_train_epochs,
            logging_dir=f"{CONFIG.output_log_dir}/logs",
            logging_strategy="steps",
            logging_steps=500,
            save_strategy="no",
            report_to="tensorboard"
        )

        # Create Trainer instance
        TRAINER = Seq2SeqTrainer(
            model=MODEL,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset["train"]
        )

        MODEL.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        #print("SAVING MODEL ..... ")
        #TRAINER.save_model(CONFIG.model_output_path)
        #TOKENIZER.save_pretrained(CONFIG.model_output_path)
        print(TRAINER.train())
        print("SAVING MODEL ..... ")
        TRAINER.save_model(CONFIG.model_output_path)
        TOKENIZER.save_pretrained(CONFIG.model_output_path)
        print("MODEL trained and saved into:", CONFIG.model_output_path)
