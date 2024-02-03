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
### 2.2. Amazon
### 2.3. ICD-10-CM
### 2.4. ACM-CCS
### 2.5. NCBI
### 2.6. Glottolog

## 3. LLMs deployment
We introduce how to deploy the LLMs used in our benchmark.

### 3.1. Llama-2s
### 3.2. Vicunas
### 3.3. Flan-t5s
### 3.4. Falcons
### 3.5. GPTs

## 4. Question generation

## 5. Evaluation
In order to conduct the experiments, please follow these steps.

### 5.1. LLama-2s
### 5.2. Vicunas
### 5.3. Flan-t5s
### 5.4. Falcons
### 5.5. GPTs
