# How to get the shopping taxonomy data
If you want to use the question pools generated directly, you can skip this page and refer to the page for [question_pools](https://github.com/ysunbp/TaxoGlimpse/tree/main/question_pools) directly!
## Step 1 Download raw data 
### Google
If you want to obtain the data from scratch: <br>
1. Please download the Google product taxonomy from [link](https://www.google.com/basepages/producttype/taxonomy.en-US.txt), put it in the ./data/ folder; <br>
2. Run the file to scrawl the instances: [google_spider_instance_typing_level.py](./scripts/google_spider_instance_typing_level.py)
### Amazon 
If you want to obtain the data from scratch: <br>
1. Please run the following files: [amazon_spider_browsenode.py](./scripts/amazon_spider_browsenode.py) and [amazon_spider_instance_typing.py](./scripts/amazon_spider_instance_typing.py) to obtain the data for main experiment and the instance typing experiment.
### eBay
Download the raw html file from [link](https://www.ebay.com/n/all-categories).
## Step 2 Generate question pools
### Google
Please follow these steps: <br>
1. Run the [google_generate_question_pool.py](./scripts/google_generate_question_pool.py) file to obtain the question pool for the main experiments.<br>
2. Run the [google_instance_typing_level_generate_question_pool.py](./scripts/google_instance_typing_level_generate_question_pool.py) file to obtain the question pool for the instance typing experiments.
### Amazon
Please follow these steps: <br>
1. Run the [amazon_generate_question_pool.py](./scripts/amazon_generate_question_pool.py) file to obtain the question pool for the main experiments.<br>
2. Run the [amazon_spider_instance_typing_level.py](./scripts/amazon_spider_instance_typing_level.py) file to obtain the question pool for the instance typing experiments.
### eBay
Please follow these steps: <be>
1. Run the [ebay_generate_question_pool.py](./scripts/ebay_generate_question_pool.py) file to obtain the question pool for the main experiments.
