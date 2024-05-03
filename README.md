# TaxoGlimpse

This is the repository for TaxoGlimpse, a benchmark evaluating LLMs' performance on taxonomies.
![benchmark-motivation](https://github.com/ysunbp/TaxoGlimpse/blob/main/figures/motivation_updated.png)

## 1. Install requirements
In order to deploy the LLMs and install requirements for the data processing scripts, we need to create the following environments: llama (for Llama-2s), vicuna-self (for vicunas), falcon (for falcons), flan-t5 (for flan-t5s), and LLM-probing (for GPTs and data processing). We now introduce how to create these environments with Anaconda.

### 1.1. llama
```console
$ conda create -n llama python=3.10
$ cd LLMs/llama
$ pip install -e .
```

### 1.2. vicuna-self
```console
$ conda create -n vicuna-self python=3.10
$ cd LLMs/vicuna/FastChat
$ pip3 install -e ".[model_worker,webui]"
```

### 1.3. falcon
```console
$ conda create -n falcon python=3.10
$ cd requirements
$ pip install -r falcon.txt
```

### 1.4. flan-t5
```console
$ conda create -n flan-t5 python=3.10
$ cd requirements
$ pip install -r flan-t5.txt
```

### 1.5. LLM-probing
```console
$ conda create -n LLM-probing python=3.10
$ cd requirements
$ pip install -r LLM-probing.txt
```

## 2. Data collection
The data collection process of the taxonomies is as follows:

### 2.1. Google
We obtained the Google Product Category taxonomy from [link](https://www.google.com/basepages/producttype/taxonomy.en-US.txt) and crawled the product instances to perform the additional instance typing experiment. For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/shopping/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/shopping).
### 2.2. Amazon
We crawled Amazon's Product Category and the product instances from the [browsenodes.com](https://www.browsenodes.com/). We provide the detailed scripts, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/shopping/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/shopping).
### 2.3. ICD-10-CM
We accessed the ICD-10-CM taxonomy through the [simple-icd-10](https://pypi.org/project/simple-icd-10/) package (version 2.0.1), for detailed usage, please refer to the [github repo](https://github.com/StefanoTrv/simple_icd_10_CM) of simple-icd-10. For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/medical/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/medical).
### 2.4. ACM-CCS
The ACM-CCS taxonomy was obtained from the following [link](https://dl.acm.org/pb-assets/dl_ccs/acm_ccs2012-1626988337597.xml). For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/academic/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/academic).
### 2.5. NCBI
The NCBI taxonomy was downloaded through the [official download page](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/). We provide the 2023 Sept version as discussed in the README.md in [TaxoGlimpse/LLM-taxonomy/biology/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/biology).
### 2.6. Glottolog
The Glottolog taxonomy (Version 4.8) was obtained from the following [link](https://glottolog.org/meta/downloads). We provide the data used by us in the README.md in [TaxoGlimpse/LLM-taxonomy/language/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/language).

## 3. LLMs deployment
We introduce how to deploy the LLMs used in our benchmark.

### 3.1. Llama-2s
Please refer to steps 3 to 5 of the Quick Start in [README.md](https://github.com/ysunbp/TaxoGlimpse/blob/main/LLMs/llama/README.md) file to download the model weights (7B-chat, 13B-chat, and 70B-chat).
### 3.2. Vicunas
Please refer to the Model Weights Section in [README.md](https://github.com/ysunbp/TaxoGlimpse/blob/main/LLMs/vicuna/FastChat/README.md) of Vicuna to download the weights for (lmsys/vicuna-7b-v1.5, lmsys/vicuna-13b-v1.5, and lmsys/vicuna-33b-v1.3).
### 3.3. Flan-t5s
Use the following Python code to deploy the LLMs:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_3b = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").cuda() # 3B
tokenizer_3b = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model_11b = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl").cuda() # 11B
tokenizer_11b = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
```
### 3.4. Falcons
Use the following Python code to deploy the LLMs:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_7b = "tiiuae/falcon-7b-instruct" # 7B
tokenizer_7b = AutoTokenizer.from_pretrained(model_7b)
model_40b = "tiiuae/falcon-40b-instruct" # 40B
tokenizer_40b = AutoTokenizer.from_pretrained(model_40b)
```
### 3.5. GPTs
Use the following Python code to deploy the LLMs:
#### Azure API
```python
import openai
openai.api_type = "azure"
openai.api_base = 'xxx'
openai.api_key = 'xxx'
openai.api_version = "2023-05-15" # azure API
def generateResponse(prompt, gpt_name):
    messages = [{"role": "user","content": prompt}]
    response = openai.ChatCompletion.create(
        engine=gpt_name,
        temperature=0,
        messages=messages
    )
    return response['choices'][0]['message']['content']
generateResponse("example", "gpt-35-turbo")
```
#### OpenAI API
```python
import openai
openai.api_base = ''
openai.api_key = ''
def generateResponse(prompt, gpt_name):
    messages = [{"role": "user","content": prompt}]
    response = openai.ChatCompletion.create(
        model=gpt_name,
        temperature=0,
        messages=messages
    )
    return response['choices'][0]['message']['content']
generateResponse("example", "gpt-4-1106-preview")
```
## 4. Question generation
We provide the generated question pools in [TaxoGlimpse/question_pools/](./question_pools), you can download them directly. Alternatively, if you want to generate the question pools from scratch, please refer to the README page for each domain under the sub-folders in [TaxoGlimpse/LLM-taxonomy](./LLM-taxonomy).
## 5. Evaluation
To conduct the experiments, please follow these steps.
### 5.1. LLama-2s
We introduce the steps for Llama-7B, Llama-13B, and Llama-70B respectively, including the main experiments and the instance typing experiment.
#### Llama-7B
```console
$ conda activate llama
$ cd TaxoGlimpse/LLMs/llama/
$ torchrun --nproc_per_node 1 evaluate_llama_taxonomy.py >> ../logs/llama-2-7b-chat/log.txt
```
#### Llama-13B
```console
$ conda activate llama
$ cd TaxoGlimpse/LLMs/llama/
$ torchrun --nproc_per_node 2 evaluate_llama_taxonomy.py >> ../logs/llama-2-13b-chat/log.txt
```
#### Llama-70B
```console
$ conda activate llama
$ cd TaxoGlimpse/LLMs/llama/
$ torchrun --nproc_per_node 8 evaluate_llama_taxonomy.py >> ../logs/llama-2-70b-chat/log.txt
```
### 5.2. Vicunas
We introduce the steps for Vicuna-7B, Vicuna-13B, and Vicuna-33B respectively, including the main experiments and the instance typing experiment.
#### Vicuna-7B
```console
$ conda activate vicuna-self
$ cd TaxoGlimpse/LLMs/vicuna/FastChat/
$ ### main experiments
$ python3 -m fastchat.serve.cli-zero-shot --model-path lmsys/vicuna-7b-v1.5 >> ../logs/vicuna-7b/zero_shot.txt # zero shot
$ python3 -m fastchat.serve.cli-few-shot --model-path lmsys/vicuna-7b-v1.5 >> ../logs/vicuna-7b/few_shot.txt # few shot
$ python3 -m fastchat.serve.cli-COT-shot --model-path lmsys/vicuna-7b-v1.5 >> ../logs/vicuna-7b/COT_shot.txt # COT
$ ### instance typing experiments
$ python3 -m fastchat.serve.cli-zero-shot-instance --model-path lmsys/vicuna-7b-v1.5 >> ../logs/vicuna-7b/zero_shot_instance.txt 
$ python3 -m fastchat.serve.cli-zero-shot-instance-full --model-path lmsys/vicuna-7b-v1.5 >> ../logs/vicuna-7b/zero_shot_instance_full.txt
```
#### Vicuna-13B
```console
$ conda activate vicuna-self
$ cd TaxoGlimpse/LLMs/vicuna/FastChat/
$ ### main experiments
$ python3 -m fastchat.serve.cli-zero-shot --model-path lmsys/vicuna-13b-v1.5 >> ../logs/vicuna-13b/zero_shot.txt # zero shot
$ python3 -m fastchat.serve.cli-few-shot --model-path lmsys/vicuna-13b-v1.5 >> ../logs/vicuna-13b/few_shot.txt # few shot
$ python3 -m fastchat.serve.cli-COT-shot --model-path lmsys/vicuna-13b-v1.5 >> ../logs/vicuna-13b/COT_shot.txt # COT
$ ### instance typing experiments
$ python3 -m fastchat.serve.cli-zero-shot-instance --model-path lmsys/vicuna-13b-v1.5 >> ../logs/vicuna-13b/zero_shot_instance.txt 
$ python3 -m fastchat.serve.cli-zero-shot-instance-full --model-path lmsys/vicuna-13b-v1.5 >> ../logs/vicuna-13b/zero_shot_instance_full.txt
```
#### Vicuna-33B
```console
$ conda activate vicuna-self
$ cd TaxoGlimpse/LLMs/vicuna/FastChat/
$ ### main experiments
$ python3 -m fastchat.serve.cli-zero-shot --model-path lmsys/vicuna-33b-v1.3 >> ../logs/vicuna-33b/zero_shot.txt # zero shot
$ python3 -m fastchat.serve.cli-few-shot --model-path lmsys/vicuna-33b-v1.3 >> ../logs/vicuna-33b/few_shot.txt # few shot
$ python3 -m fastchat.serve.cli-COT-shot --model-path lmsys/vicuna-33b-v1.3 >> ../logs/vicuna-33b/COT_shot.txt # COT
$ ### instance typing experiments
$ python3 -m fastchat.serve.cli-zero-shot-instance --model-path lmsys/vicuna-33b-v1.3 >> ../logs/vicuna-33b/zero_shot_instance.txt 
$ python3 -m fastchat.serve.cli-zero-shot-instance-full --model-path lmsys/vicuna-33b-v1.3 >> ../logs/vicuna-33b/zero_shot_instance_full.txt
```
### 5.3. Flan-t5s
We introduce the steps for Flan-T5-3B and Flan-T5-11B respectively, including the main experiments and the instance typing experiment.
#### Flan-T5-3B and 11B
```console
$ conda activate flan-t5
$ cd TaxoGlimpse/LLMs/flan-t5
$ ### main experiments
$ python flan_chat_taxonomy.py >> ../logs/flan-t5/zero_shot.txt # zero shot
$ python flan_chat_taxonomy_few_shot.py >> ../logs/flan-t5/few_shot.txt # few shot
$ python flan_chat_taxonomy_COT.py >> ../logs/flan-t5/COT.txt # COT
$ ### instance typing experiments
$ python flan_chat_taxonomy_instance.py >> ../logs/flan-t5/zero_shot_instance.txt
$ python flan_chat_taxonomy_instance_full.py >> ../logs/flan-t5/zero_shot_instance_full.txt
```
### 5.4. Falcons
We introduce the steps for Falcon-7B and Falcon-40B respectively, including the main experiments and the instance typing experiment.
#### Falcon-7B
```console
$ conda activate falcon
$ cd TaxoGlimpse/LLMs/falcon/7B
$ ### main experiments
$ python falcon_chat_taxonomy.py >> ../logs/falcon-7b/zero_shot.txt # zero shot
$ python falcon_chat_taxonomy_few_shot.py >> ../logs/falcon-7b/few_shot.txt # few shot
$ python falcon_chat_taxonomy_COT.py >> ../logs/falcon-7b/COT.txt # COT
$ ### instance typing experiments
$ python falcon_chat_taxonomy_instance.py >> ../logs/falcon-7b/zero_shot_instance.txt 
$ python falcon_chat_taxonomy_instance_full.py >> ../logs/falcon-7b/zero_shot_instance_full.txt
```
#### Falcon-40B
```console
$ conda activate falcon
$ cd TaxoGlimpse/LLMs/falcon/40B
$ ### main experiments
$ python falcon_chat_taxonomy.py >> ../logs/falcon-40b/zero_shot.txt # zero shot
$ python falcon_chat_taxonomy_few_shot.py >> ../logs/falcon-40b/few_shot.txt # few shot
$ python falcon_chat_taxonomy_COT.py >> ../logs/falcon-40b/COT.txt # COT
$ ### instance typing experiments
$ python falcon_chat_taxonomy_instance.py >> ../logs/falcon-40b/zero_shot_instance.txt 
$ python falcon_chat_taxonomy_instance_full.py >> ../logs/falcon-40b/zero_shot_instance_full.txt
``` 
### 5.5. GPTs
We introduce the steps for GPT-3.5 and GPT-4 respectively, including the main experiments and the instance typing experiment. <br>
Please input your Azure APIs or OpenAI APIs at the beginning of the Python files.
#### GPT 3.5
```console
$ conda activate LLM-probing
$ cd TaxoGlimpse/LLMs/GPTs
$ ### main experiments
$ evaluate_taxonomy.py >> ../logs/gpt-3.5/main_exps.txt # zero shot
$ ### instance typing experiments
$ evaluate_taxonomy_instance.py >> ../logs/gpt-3.5/instance.txt
$ evaluate_taxonomy_instance_full.py >> ../logs/gpt-3.5/instance_full.txt
```
#### GPT 4
```console
$ conda activate LLM-probing
$ cd TaxoGlimpse/LLMs/GPTs
$ ### main experiments
$ evaluate_taxonomy_gpt4.py >> ../logs/gpt-4/main_exps.txt # zero shot
$ ### instance typing experiments
$ evaluate_taxonomy_gpt4_instance.py >> ../logs/gpt-4/instance.txt
$ evaluate_taxonomy_gpt4_instance_full.py >> ../logs/gpt-4/instance_full.txt
```
## 6. Main results
![results](https://github.com/ysunbp/TaxoGlimpse/blob/main/figures/result.png)
