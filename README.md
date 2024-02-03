# TaxoGlimpse

This is the repository for TaxoGlimpse, a benchmark evaluating LLMs' performance on taxonomies.

## 1. Install requirements
In order to deploy the LLMs and install requirements for the data processing scripts, we need to create the following environments: llama (for Llama-2s), vicuna-self (for vicunas), falcon (for falcons), flan-t5 (for flan-t5s), and LLM-probing (for GPTs and data processing). We now introduce how to create these environments with Anaconda.

## 1.1. llama
```console
$ conda create -n llama python=3.10
$ cd LLMs/llama
$ pip install -e .
```

## 1.2. vicuna-self
```console
$ conda create -n vicuna-self python=3.10
$ cd LLMs/vicuna/FastChat
$ pip3 install -e ".[model_worker,webui]"
```

## 1.3. falcon
```console
$ conda create -n falcon python=3.10
$ cd requirements
$ pip install -r falcon.txt
```

## 1.4. flan-t5
```console
$ conda create -n flan-t5 python=3.10
$ cd requirements
$ pip install -r flan-t5.txt
```

## 1.5. LLM-probing
```console
$ conda create -n LLM-probing python=3.10
$ cd requirements
$ pip install -r LLM-probing.txt
```

## 2. Data collection

## 3. LLMs deployment

## 4. Question generation

## 5. Evaluation
