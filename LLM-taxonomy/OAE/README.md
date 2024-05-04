# How to get the OAE taxonomy data
If you want to use the question pools generated directly, you can skip this page and refer to the page for [question_pools](https://github.com/ysunbp/TaxoGlimpse/tree/main/question_pools) directly!
## Generate question pools
Please follow these steps: <br>
1. Download the raw file from [link](https://bioportal.bioontology.org/ontologies/OAE)
2. Run the [OAE_generate_question_pool.py](./scripts/OAE_generate_question_pool.py) file to obtain the question pool for the main experiments.
3. Run the [OAE_generate_question_pool_mcq.py](./scripts/OAE_generate_question_pool_mcq.py) file to obtain the question pool for the MCQ experiments.
4. Run the [OAE_generate_question_pool_instance.py](./scripts/OAE_generate_question_pool_instance.py) file to obtain the question pool for the instance typing experiments.
