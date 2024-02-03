# TaxoGlimpse

This is the repository for TaxoGlimpse, a benchmark evaluating LLMs' performance on taxonomies.

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
We obtained the Google Product Category taxonomy from [link](https://www.google.com/basepages/producttype/taxonomy.en-US.txt) and scrawled the product instances to perform the additional instance typing experiment. For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/shopping/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/shopping).
### 2.2. Amazon
We crawled Amazon's Product Category and the product instances from the [browsenodes.com](https://www.browsenodes.com/). We provide the detailed scripts, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/shopping/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/shopping).
### 2.3. ICD-10-CM
We accessed the ICD-10-CM taxonomy through the [simple-icd-10](https://pypi.org/project/simple-icd-10/) package (version 2.0.1), for detailed usage, please refer to the [github repo](https://github.com/StefanoTrv/simple_icd_10_CM) of simple-icd-10. For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/shopping/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/medical).
### 2.4. ACM-CCS
The ACM-CCS taxonomy was obtained from the following [link](https://dl.acm.org/pb-assets/dl_ccs/acm_ccs2012-1626988337597.xml). For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/shopping/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/academic).
### 2.5. NCBI
The NCBI taxonomy was downloaded through the [official download page](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/). We provide the 2023 Sept version as discussed in the README.md in [TaxoGlimpse/LLM-taxonomy/biology/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/biology).
### 2.6. Glottolog
The Glottolog taxonomy (Version 4.8) was obtained from the following [link](https://glottolog.org/meta/downloads). We provide the data used by us in the README.md in [TaxoGlimpse/LLM-taxonomy/language/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/language).

## 3. LLMs deployment
We introduce how to deploy the LLMs used in our benchmark.

### 3.1. Llama-2s
Please refer to steps 3 to 5 of the Quick Start in [README.md](https://github.com/ysunbp/TaxoGlimpse/blob/main/LLMs/llama/README.md) file to download the model weights (7B-chat, 13B-caht, and 70B-chat).
### 3.2. Vicunas
Please refer to the Model Weights Section in [README.md](https://github.com/ysunbp/TaxoGlimpse/blob/main/LLMs/vicuna/FastChat/README.md) of Vicuna to download the weights for (lmsys/vicuna-7b-v1.5, lmsys/vicuna-13b-v1.5, and lmsys/vicuna-33b-v1.3).
### 3.3. Flan-t5s
Use the following python code to deploy the LLMs:
```
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_3b = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").cuda() # 3B
tokenizer_3b = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model_11b = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl").cuda() # 11B
tokenizer_11b = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
```
### 3.4. Falcons
```
from transformers import AutoTokenizer, AutoModelForCausalLM

model_7b = "tiiuae/falcon-7b-instruct" # 7B
tokenizer_7b = AutoTokenizer.from_pretrained(model_7b)
model_40b = "tiiuae/falcon-40b-instruct" # 40B
tokenizer_40b = AutoTokenizer.from_pretrained(model_40b)
```
### 3.5. GPTs
#### GPT-3.5
```
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
#### GPT-4
```
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

## 5. Evaluation
In order to conduct the experiments, please follow these steps.

### 5.1. LLama-2s
### 5.2. Vicunas
### 5.3. Flan-t5s
### 5.4. Falcons
### 5.5. GPTs
