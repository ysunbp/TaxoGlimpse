# How to get the raw biology taxonomy data
If you want to use the question pools generated directly, you can skip this page and refer to the page for [question_pools](https://github.com/ysunbp/TaxoGlimpse/tree/main/question_pools) directly!
## Step 1 Download raw data 
Please download the raw biology taxonomy data from the following [link](https://drive.google.com/file/d/1R3614fV3xhXBDEopX_bNKUyM3F1bME8K/view?usp=drive_link); We also provide the temperary [files](https://drive.google.com/file/d/1-EcyUFYXRa0uJsniWaMaRwnzXMwub-Hn/view?usp=drive_link). You can download and unzip these files and put them in the biology folder.
## Step 2 Generate question pools
Please follow these steps: <br>
1. Run the [biology_generate_question_pool.py](./scripts/biology_generate_question_pool.py) file to obtain the question pool for the main experiments.<br>
2. Run the [biology_generate_question_pool_instance.py](./scripts/biology_generate_question_pool_instance.py) file to obtain the question pool for the instance typing experiments.
