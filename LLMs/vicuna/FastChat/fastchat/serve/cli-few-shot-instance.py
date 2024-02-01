"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0

Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""
import argparse
import os
import re
import sys
import csv
import random
from tqdm import tqdm

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
import torch

from fastchat.model.model_adapter import add_model_args, load_model
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.serve.inference import ChatIO, chat_loop, chat_question_pool_few_shot
from fastchat.utils import str_to_torch_dtype





class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        return
        #print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            #print('stream',output_text)
            if now > pre:
                #print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        #print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

    def print_output(self, text: str):
        print(text)


class RichChatIO(ChatIO):
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event):
        event.app.current_buffer.newline()

    def __init__(self, multiline: bool = False, mouse: bool = False):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(
            words=["!!exit", "!!reset", "!!remove", "!!regen", "!!save", "!!load"],
            pattern=re.compile("$"),
        )
        self._console = Console()
        self._multiline = multiline
        self._mouse = mouse

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            mouse_support=self._mouse,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.bindings if self._multiline else None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role.replace('/', '|')}:")

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return text

    def print_output(self, text: str):
        self.stream_output([{"text": text}])


class ProgrammaticChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        contents = ""
        # `end_sequence` signals the end of a message. It is unlikely to occur in
        #  message content.
        end_sequence = " __END_OF_A_MESSAGE_47582648__\n"
        len_end = len(end_sequence)
        while True:
            if len(contents) >= len_end:
                last_chars = contents[-len_end:]
                if last_chars == end_sequence:
                    break
            try:
                char = sys.stdin.read(1)
                contents = contents + char
            except EOFError:
                continue
        contents = contents[:-len_end]
        print(f"[!OP:{role}]: {contents}", flush=True)
        return contents

    def prompt_for_output(self, role: str):
        print(f"[!OP:{role}]: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

    def print_output(self, text: str):
        print(text)

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

def sample_few_shot_examples(cur_question_pool_dicts, question_bank, exp_type, n=5):
    ### cur_question_pool_dict: {key: level_{level}_question_type; value: [(parent, child),(parent, child),...]}
    out_example_dict = {}

    for cur_question_pool_dict in cur_question_pool_dicts:
        sample_key = list(cur_question_pool_dict.keys())[0]
        if exp_type == 'few-shot':
            sample_type = sample_key[8:]
        else:
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

    sampled_few_shots = sample_few_shot_examples(sampled_question_pairs_list, question_bank, exp_type='instance-few', n=5)
    return sampled_question_pairs, sampled_few_shots

### copied from evaluate_llama_taxonomy.py

def get_question_pool(sampled_question_pairs, taxonomy_type):
    # TODO: Write this function after we generate the question pool, input is the path, output is the set of questions to be asked. remember to run pip3 install -e ".[model_worker,webui]" at the root directory in order to update the package.
    cur_sampled_question_pairs, sampled_few_shot = sampled_question_pairs
    clue = 'Always answer with brief answers Yes, No, or I don\'t know. '
    print('current taxonomy type:', taxonomy_type)
    out_dict = {}
    for sampled_question_key in cur_sampled_question_pairs.keys():
        cur_question_cat = sampled_question_key
        for idx, (roottype, subtype) in enumerate(tqdm(cur_sampled_question_pairs[sampled_question_key])):
            dialog = {}
            cur_examples = sampled_few_shot[sampled_question_key][idx]
            cur_num_positive = cur_examples[-1]
            cur_positive_examples = cur_examples[:cur_num_positive]
            cur_negative_examples = cur_examples[cur_num_positive:-1]
            dialog['message'] = []
            if taxonomy_type == 'academic-acm':
                dialog['system'] = clue 
                dialog['user'] = 'Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)'
                for roottype, subtype in cur_positive_examples:
                    dialog['message'].append(('Example: Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)','Yes.'))
                for roottype, subtype in cur_negative_examples:
                    dialog['message'].append(('Example: Is ' + subtype + ' computer science research concept a type of '+roottype +' computer science research concept? answer with (Yes/No/I don\'t know)','No.'))
            elif taxonomy_type == 'biology-NCBI':
                dialog['system'] = clue
                dialog['user'] = 'Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)'
                for roottype, subtype in cur_positive_examples:
                    dialog['message'].append(('Example: Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)','Yes.'))
                for roottype, subtype in cur_negative_examples:
                    dialog['message'].append(('Example: Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)','No.'))
            elif taxonomy_type == 'language-glottolog':
                dialog['system'] = clue 
                dialog['user'] = 'Is ' + subtype + ' language a type of '+roottype +' language? answer with (Yes/No/I don\'t know)'
                for roottype, subtype in cur_positive_examples:
                    dialog['message'].append(('Example: Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)','Yes.'))
                for roottype, subtype in cur_negative_examples:
                    dialog['message'].append(('Example: Is ' + subtype + ' a type of '+roottype +'? answer with (Yes/No/I don\'t know)','No.'))
            elif taxonomy_type == 'medical-icd':
                dialog['system'] = clue 
                dialog['user'] = 'Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)'
                for roottype, subtype in cur_positive_examples:
                    dialog['message'].append(('Example: Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)','Yes.'))
                for roottype, subtype in cur_negative_examples:
                    dialog['message'].append(('Example: Is ' + subtype.lower() + ' a type of '+roottype.lower() +'? answer with (Yes/No/I don\'t know)','No.'))
            elif taxonomy_type == 'shopping-amazon':
                dialog['system'] = clue 
                dialog['user'] = 'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)'
                for roottype, subtype in cur_positive_examples:
                    dialog['message'].append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','Yes.'))
                for roottype, subtype in cur_negative_examples:
                    dialog['message'].append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','No.'))
            elif taxonomy_type == 'shopping-google':
                dialog['system'] = clue 
                dialog['user'] = 'Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)'
                for roottype, subtype in cur_positive_examples:
                    dialog['message'].append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','Yes.'))
                for roottype, subtype in cur_negative_examples:
                    dialog['message'].append(('Example: Are ' + subtype + ' products a type of '+roottype +' products? answer with (Yes/No/I don\'t know)','No.'))
            
            if not cur_question_cat in out_dict.keys():
                out_dict[cur_question_cat] = [dialog]
            else:
                out_dict[cur_question_cat].append(dialog)
    return out_dict

def main(args):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["XPU_VISIBLE_DEVICES"] = args.gpus
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        if args.question_path:
            model, tokenizer = load_model(
                args.model_path,
                device=args.device,
                num_gpus=args.num_gpus,
                max_gpu_memory=args.max_gpu_memory,
                dtype=str_to_torch_dtype(args.dtype),
                load_8bit=args.load_8bit,
                cpu_offloading=args.cpu_offloading,
                gptq_config=GptqConfig(
                                    ckpt=args.gptq_ckpt or args.model_path,
                                    wbits=args.gptq_wbits,
                                    groupsize=args.gptq_groupsize,
                                    act_order=args.gptq_act_order,
                                ),
                awq_config=AWQConfig(
                                    ckpt=args.awq_ckpt or args.model_path,
                                    wbits=args.awq_wbits,
                                    groupsize=args.awq_groupsize,
                                ),
                exllama_config=exllama_config,
                xft_config=xft_config,
                revision=args.revision,
                debug=args.debug,
                )
            for cur_taxonomy in args.question_path:
                cur_question_pairs = []
                sampled_question_pairs = get_sampled_pairs(sub_question_type='level', level_question_types=['positive', 'negative_hard', 'negative_easy', 'positive_to_root', 'negative_to_root'], toroot_question_types=None, question_pool_name = cur_taxonomy)
                cur_question_pairs.append(sampled_question_pairs)
                for cur_sampled_question_pairs in cur_question_pairs:
                    inps_dict = get_question_pool(cur_sampled_question_pairs, cur_taxonomy)
                
                    for cur_inps_key in inps_dict.keys():
                        inps_message_list = inps_dict[cur_inps_key]
                        cur_question_cat = cur_inps_key
                        cur_path = 'TaxoGlimpse/results/instance-few/'+cur_taxonomy+'/'+args.model_path.split('/')[1]+'/'+cur_question_cat+'.csv'
                        yes_total = 0
                        no_total = 0
                        dont_total = 0
                        with open(cur_path, 'a', newline='') as file:
                            csv_writer = csv.writer(file)
                            #inps = ["Always answer with brief answers Yes, No, or I don't know. Question: Are Piano Blues products a type of Blues products? answer with (Yes/No/I don't know)", "Always answer with brief answers Yes, No, or I don't know. Question: Are Piano Blues products a type of Blues products? answer with (Yes/No/I don't know)"]
                            outs = chat_question_pool_few_shot(
                                inps_message_list,
                                args.model_path,
                                model,
                                tokenizer,
                                args.device,
                                args.conv_template,
                                args.conv_system_msg,
                                args.temperature,
                                args.repetition_penalty,
                                args.max_new_tokens,
                                chatio,
                                judge_sent_end=args.judge_sent_end,
                                history=not args.no_history,
                            )
                            for idx, item in enumerate(tqdm(outs)):
                                if 'yes' in item.strip().lower():
                                    decision = 1
                                    yes_total += 1
                                elif 'know' in item.strip().lower():
                                    decision = 2
                                    dont_total += 1
                                else:
                                    decision = 0
                                    no_total += 1
                                root, sub = cur_sampled_question_pairs[0][cur_question_cat][idx] # for few shot [0] is the current sample dict, [1] is its few shot examples
                                cur_row = (root, sub, item, decision)
                                csv_writer.writerow(cur_row)
                        total_num_questions = len(outs)
                        if 'positive' in cur_question_cat:
                            acc = yes_total/total_num_questions
                        else:
                            acc = no_total/total_num_questions
                        miss_rate = dont_total/total_num_questions
                        print('summary of exp:', cur_question_cat, ', total number of questions:', total_num_questions, ', yes total:', yes_total, ', no total:', no_total, ', miss total:', dont_total, 'acc', acc, 'missing rate', miss_rate)
                        print('++++++++++++++++++++++++++++++++++++++++')
        else:
            chat_loop(
                args.model_path,
                args.device,
                args.num_gpus,
                args.max_gpu_memory,
                str_to_torch_dtype(args.dtype),
                args.load_8bit,
                args.cpu_offloading,
                args.conv_template,
                args.conv_system_msg,
                args.temperature,
                args.repetition_penalty,
                args.max_new_tokens,
                chatio,
                gptq_config=GptqConfig(
                    ckpt=args.gptq_ckpt or args.model_path,
                    wbits=args.gptq_wbits,
                    groupsize=args.gptq_groupsize,
                    act_order=args.gptq_act_order,
                ),
                awq_config=AWQConfig(
                    ckpt=args.awq_ckpt or args.model_path,
                    wbits=args.awq_wbits,
                    groupsize=args.awq_groupsize,
                ),
                exllama_config=exllama_config,
                xft_config=xft_config,
                revision=args.revision,
                judge_sent_end=args.judge_sent_end,
                debug=args.debug,
                history=not args.no_history,
            )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-history", action="store_true")

    parser.add_argument("--question_path", type=list, default=['shopping-amazon'], choices=['academic-acm', 'biology-NCBI', 'language-glottolog', 'medical-icd', 'shopping-amazon', 'shopping-google'])

    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--judge-sent-end",
        action="store_true",
        help="Whether enable the correction logic that interrupts the output of sentences due to EOS.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)
    ### command: python3 -m fastchat.serve.cli-few-shot-instance --model-path lmsys/vicuna-7b-v1.5 >> /export/data/LLM-benchmark-project-KB/logs/vicuna-7b/instance-amazon-few-shot.txt
    ### command: python3 -m fastchat.serve.cli-few-shot-instance --model-path lmsys/vicuna-13b-v1.5 >> /export/data/LLM-benchmark-project-KB/logs/vicuna-13b/instance-amazon-few-shot.txt
    ### command: python3 -m fastchat.serve.cli-few-shot-instance --model-path lmsys/vicuna-33b-v1.3 >> /export/data/LLM-benchmark-project-KB/logs/vicuna-33b/instance-amazon-few-shot.txt