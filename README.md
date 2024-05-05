# TaxoGlimpse

This is the repository for TaxoGlimpse, a benchmark evaluating LLMs' performance on taxonomies.
![benchmark-motivation](https://github.com/ysunbp/TaxoGlimpse/blob/main/figures/motivation_updated.png)

## 1. Install requirements
In order to deploy the LLMs and install requirements for the data processing scripts, we need to create the following environments: llama (for Llama-2s), vicuna-self (for vicunas), falcon (for falcons), flan-t5 (for flan-t5s), LLM-probing (for GPTs, Claude-3 and data processing), mixtral (for mistral and mixtral), llama3 (for Llama-3s), and llms4ol (for LLMs4OL). We now introduce how to create these environments with Anaconda.

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

### 1.6. mixtral
```console
$ conda create -n mixtral python=3.8
$ cd requirements
$ pip install -r mixtral.txt
```

### 1.7. llama3
```console
$ conda create -n llama3 python=3.10
$ cd requirements
$ pip install -r llama3.txt
```

### 1.8. llms4ol
```console
$ conda create -n llms4ol python=3.9
$ cd requirements
$ pip install -r llms4ol.txt
```

## 2. Data collection
The data collection process of the taxonomies is as follows:

### 2.1. eBay
We crawled the eBay taxonomy from [link](https://www.ebay.com/n/all-categories). For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/shopping/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/shopping).
### 2.2. Google
We obtained the Google Product Category taxonomy from [link](https://www.google.com/basepages/producttype/taxonomy.en-US.txt) and crawled the product instances to perform the additional instance typing experiment. For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/shopping/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/shopping).
### 2.3. Amazon
We crawled Amazon's Product Category and the product instances from the [browsenodes.com](https://www.browsenodes.com/). We provide the detailed scripts, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/shopping/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/shopping).
### 2.4. Schema.org
We downloaded the Schema.org data from [link](https://github.com/schemaorg/schemaorg/blob/main/data/releases/26.0/schemaorg-current-http-types.csv). For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/general/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/general).
### 2.5. ACM-CCS
The ACM-CCS taxonomy was obtained from the following [link](https://dl.acm.org/pb-assets/dl_ccs/acm_ccs2012-1626988337597.xml). For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/academic/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/academic).
### 2.6. GeoNames
We download the GeoNames data from [link](https://www.geonames.org/export/codes.html). For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/geography/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/geography).
### 2.7. Glottolog
The Glottolog taxonomy (Version 4.8) was obtained from the following [link](https://glottolog.org/meta/downloads). We provide the data used by us in the README.md in [TaxoGlimpse/LLM-taxonomy/language/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/language).
### 2.8. ICD-10-CM
We accessed the ICD-10-CM taxonomy through the [simple-icd-10](https://pypi.org/project/simple-icd-10/) package (version 2.0.1), for detailed usage, please refer to the [github repo](https://github.com/StefanoTrv/simple_icd_10_CM) of simple-icd-10. For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/medical/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/medical).
### 2.9. OAE
We download the OAE taxonomy from the [link](https://bioportal.bioontology.org/ontologies/OAE). For details, please refer to the README.md in [TaxoGlimpse/LLM-taxonomy/OAE/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/OAE).
### 2.10. NCBI
The NCBI taxonomy was downloaded through the [official download page](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/). We provide the 2023 Sept version as discussed in the README.md in [TaxoGlimpse/LLM-taxonomy/biology/](https://github.com/ysunbp/TaxoGlimpse/tree/main/LLM-taxonomy/biology).


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
from openai import AzureOpenAI
client = AzureOpenAI(
        azure_endpoint = 'https://hkust.azure-api.net',
        api_key = 'xxxxx',
        api_version = "2023-05-15"
    )
def generateResponse(prompt, gpt_name):
    messages = [{"role": "user","content": prompt}]
    response = client.chat.completions.create(
        model=gpt_name,
        temperature=0,
        messages=messages
    )
    return response.choices[0].message.content
generateResponse("example", "gpt-35-turbo")
```
#### OpenAI API
```python
from openai import OpenAI
client = OpenAI(
    base_url = 'xxxx',
    api_key = 'xxxx'
)
def generateResponse(prompt, gpt_name):
    messages = [{"role": "user","content": prompt}]
    response = client.chat.completions.create(
        model=gpt_name,
        temperature=0,
        messages=messages
    )
    return response.choices[0].message.content
generateResponse("example", "gpt-4-1106-preview")
```
### 3.6. Claude
```python
import os
from litellm import completion
os.environ["ANTHROPIC_API_KEY"] = "XXX"

def generateResponse(prompt):
    messages = [{"role": "user","content": prompt['user']}]
    response = completion(model="claude-3-opus-20240229", messages=messages, api_base="https://api.openai-proxy.org/anthropic/v1/messages", temperature=0)
    return response['choices'][0]['message']['content']

generateResponse("example")
```
### 3.7. Llama-3s
Please refer to [README.md](https://github.com/meta-llama/llama3) for a quick start.
### 3.8. Mistral and Mixtral
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_mistral = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").cuda()
tokenizer_mistral = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model_mixtral = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto")
tokenizer_mixtral = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
```
### 3.9. LLMs4OL
You can use the same code for Flan-T5-3B to deploy the model, by modifying the model weights path.

## 4. Question generation
You can generate the question pools from scratch by referring to the README page for each domain under the sub-folders in [TaxoGlimpse/LLM-taxonomy](./LLM-taxonomy).
## 5. Evaluation
To conduct the experiments, please follow these steps.
### 5.1. LLama-2s
We introduce the steps for Llama-7B, Llama-13B, and Llama-70B respectively, including the main experiments and the instance typing experiment.
#### Llama-2-7B
```console
$ conda activate llama
$ cd TaxoGlimpse/LLMs/llama/
$ torchrun --nproc_per_node 1 tf-variants.py >> ../logs/llama-2-7b-chat/tf-log.txt
$ torchrun --nproc_per_node 1 mcq-variants.py >> ../logs/llama-2-7b-chat/mcq-log.txt
$ ### instance typing experiment
$ torchrun --nproc_per_node 1 instance.py >> ../logs/llama-2-7b-chat/instance-log.txt
```
#### Llama-2-13B
```console
$ conda activate llama
$ cd TaxoGlimpse/LLMs/llama/
$ torchrun --nproc_per_node 2 tf-variants.py >> ../logs/llama-2-13b-chat/tf-log.txt
$ torchrun --nproc_per_node 2 mcq-variants.py >> ../logs/llama-2-13b-chat/mcq-log.txt
$ ### instance typing experiment
$ torchrun --nproc_per_node 2 instance.py >> ../logs/llama-2-13b-chat/instance-log.txt
```
#### Llama-2-70B
```console
$ conda activate llama
$ cd TaxoGlimpse/LLMs/llama/
$ torchrun --nproc_per_node 8 tf-variants.py >> ../logs/llama-2-70b-chat/tf-log.txt
$ torchrun --nproc_per_node 8 mcq-variants.py >> ../logs/llama-2-70b-chat/mcq-log.txt
$ ### instance typing experiment
$ torchrun --nproc_per_node 8 instance.py >> ../logs/llama-2-70b-chat/instance.txt
```
### 5.2. Vicunas
We introduce the steps for Vicuna-7B, Vicuna-13B, and Vicuna-33B respectively, including the main experiments and the instance typing experiment.
#### Vicuna-7B
```console
$ conda activate vicuna-self
$ cd TaxoGlimpse/LLMs/vicuna/FastChat/
$ ### main experiments
$ python3 -m fastchat.serve.tf-variants --model-path lmsys/vicuna-7b-v1.5 >> ../logs/vicuna-7b/tf-log.txt
$ python3 -m fastchat.serve.mcq-variants --model-path lmsys/vicuna-7b-v1.5 >> ../logs/vicuna-7b/mcq-log.txt
$ ### instance typing experiments
$ python3 -m fastchat.serve.instance --model-path lmsys/vicuna-7b-v1.5 >> ../logs/vicuna-7b/instance-log.txt 
```
#### Vicuna-13B
```console
$ conda activate vicuna-self
$ cd TaxoGlimpse/LLMs/vicuna/FastChat/
$ ### main experiments
$ python3 -m fastchat.serve.tf-variants --model-path lmsys/vicuna-13b-v1.5 >> ../logs/vicuna-13b/tf-log.txt
$ python3 -m fastchat.serve.mcq-variants --model-path lmsys/vicuna-13b-v1.5 >> ../logs/vicuna-13b/mcq-log.txt
$ ### instance typing experiments
$ python3 -m fastchat.serve.instance --model-path lmsys/vicuna-13b-v1.5 >> ../logs/vicuna-13b/instance-log.txt 
```
#### Vicuna-33B
```console
$ conda activate vicuna-self
$ cd TaxoGlimpse/LLMs/vicuna/FastChat/
$ ### main experiments
$ python3 -m fastchat.serve.tf-variants --model-path lmsys/vicuna-33b-v1.3 >> ../logs/vicuna-33b/tf-log.txt
$ python3 -m fastchat.serve.mcq-variants --model-path lmsys/vicuna-33b-v1.3 >> ../logs/vicuna-33b/mcq-log.txt
$ ### instance typing experiments
$ python3 -m fastchat.serve.instance --model-path lmsys/vicuna-33b-v1.3 >> ../logs/vicuna-33b/instance-log.txt 
```
### 5.3. Flan-t5s
We introduce the steps for Flan-T5-3B and Flan-T5-11B respectively, including the main experiments and the instance typing experiment.
#### Flan-T5-3B and 11B
```console
$ conda activate flan-t5
$ cd TaxoGlimpse/LLMs/flan-t5
$ ### main experiments
$ python tf-variants.py >> ../logs/flan-t5/tf-log.txt
$ python mcq-variants.py >> ../logs/flan-t5/mcq-log.txt
$ ### instance typing experiments
$ python instance.py >> ../logs/flan-t5/instance-log.txt
```
### 5.4. Falcons
We introduce the steps for Falcon-7B and Falcon-40B respectively, including the main experiments and the instance typing experiment.
#### Falcon-7B
```console
$ conda activate falcon
$ cd TaxoGlimpse/LLMs/falcon/7B
$ ### main experiments
$ python tf-variants.py >> ../logs/falcon-7b/tf-log.txt
$ python mcq-variants.py >> ../logs/falcon-7b/mcq-log.txt
$ ### instance typing experiments
$ python instance.py >> ../logs/falcon-7b/instance-log.txt 
```
#### Falcon-40B
```console
$ conda activate falcon
$ cd TaxoGlimpse/LLMs/falcon/40B
$ ### main experiments
$ python tf-variants.py >> ../logs/falcon-40b/tf-log.txt
$ python mcq-variants.py >> ../logs/falcon-40b/mcq-log.txt
$ ### instance typing experiments
$ python instance.py >> ../logs/falcon-40b/instance-log.txt 
```
### 5.5. GPTs
We introduce the steps for GPT-3.5 and GPT-4 respectively, including the main experiments and the instance typing experiment. <br>
Please input your Azure APIs or OpenAI APIs at the beginning of the Python files.
#### GPT 3.5
```console
$ conda activate LLM-probing
$ cd TaxoGlimpse/LLMs/GPT3.5
$ ### main experiments
$ python tf-variants.py >> ../logs/gpt-3.5/tf-log.txt
$ python mcq-variants.py >> ../logs/gpt-3.5/mcq-log.txt 
$ ### instance typing experiments
$ python instance.py >> ../logs/gpt-3.5/instance-log.txt 
```
#### GPT 4
```console
$ conda activate LLM-probing
$ cd TaxoGlimpse/LLMs/GPT4
$ ### main experiments
$ python tf-variants.py >> ../logs/gpt-4/tf-log.txt
$ python mcq-variants.py >> ../logs/gpt-4/mcq-log.txt 
$ ### instance typing experiments
$ python instance.py >> ../logs/gpt-4/instance-log.txt 
```
### 5.6. Claude
We introduce the steps for Claude-3, including the main experiments and the instance typing experiment. <br>
Please input your Anthropic APIs at the beginning of the Python files.
#### Claude 3
```console
$ conda activate LLM-probing
$ cd TaxoGlimpse/LLMs/Claude
$ ### main experiments
$ python tf-variants.py >> ../logs/Claude/tf-log.txt
$ python mcq-variants.py >> ../logs/Claude/mcq-log.txt 
$ ### instance typing experiments
$ python instance.py >> ../logs/Claude/instance-log.txt 
```
### 5.7. Llama-3s
We introduce the steps for Llama-3-8B and Llama-3-70B respectively, including the main experiments and the instance typing experiment.
#### Llama-3-8B
```console
$ conda activate llama3
$ cd TaxoGlimpse/LLMs/llama3/
$ torchrun --nproc_per_node 1 tf-variants.py >> ../logs/llama-3-8b/tf-log.txt
$ torchrun --nproc_per_node 1 mcq-variants.py >> ../logs/llama-3-8b/mcq-log.txt
$ ### instance typing experiment
$ torchrun --nproc_per_node 1 instance.py >> ../logs/llama-3-8b/instance-log.txt
```
#### Llama-3-70B
```console
$ conda activate llama3
$ cd TaxoGlimpse/LLMs/llama3/
$ torchrun --nproc_per_node 8 tf-variants.py >> ../logs/llama-3-70b/tf-log.txt
$ torchrun --nproc_per_node 8 mcq-variants.py >> ../logs/llama-3-70b/mcq-log.txt
$ ### instance typing experiment
$ torchrun --nproc_per_node 8 instance.py >> ../logs/llama-3-70b/instance.txt
```
### 5.8. Mistral and Mixtral
We introduce the steps for Mistral and Mixtral respectively, including the main experiments and the instance typing experiment.
#### Mistral
```console
$ conda activate mixtral
$ cd TaxoGlimpse/LLMs/Mistral-Mixtral/
$ python tf-variants.py >> ../logs/mistral/tf-log.txt
$ python mcq-variants.py >> ../logs/mistral/mcq-log.txt 
$ ### instance typing experiments
$ python instance.py >> ../logs/mistral/instance-log.txt 
```
#### Mixtral
```console
$ conda activate mixtral
$ cd TaxoGlimpse/LLMs/Mistral-Mixtral/
$ python tf-variants.py >> ../logs/mixtral/tf-log.txt
$ python mcq-variants.py >> ../logs/mixtral/mcq-log.txt 
$ ### instance typing experiments
$ python instance.py >> ../logs/mixtral/instance-log.txt 
```
### 5.9. LLMs4OL
We introduce the steps for LLMs4OL, including the main experiments and the instance typing experiment.
```console
$ conda activate llms4ol
$ cd TaxoGlimpse/LLMs/LLMs4OL/tuning
$ ### instruction tuning for main experiments
$ python3 trainer.py
$ ### instruction tuning for instance typing experiments
$ python3 trainer-instance.py
$ cd TaxoGlimpse/LLMs/LLMs4OL/taxoglimpse
$ ### main experiments
$ python tf-variants.py >> ../logs/llms4ol/tf-log.txt
$ python mcq-variants.py >> ../logs/llms4ol/mcq-log.txt 
$ ### instance typing experiments
$ python instance.py >> ../logs/llms4ol/instance-log.txt 
```
## 6. Main results
![results](https://github.com/ysunbp/TaxoGlimpse/blob/main/figures/result.png)
