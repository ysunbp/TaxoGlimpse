from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from fake_useragent import UserAgent
import pickle
import os
import time
import random

def clean_name(string):
    return string.replace('&amp;', '&')


ua = UserAgent()
cur_ua = ua.random
cur_headers = {'content-type': 'text/html; charset=UTF-8',
    'user-agent': cur_ua,
    'Referer': 'https://www.google.com/',
    'Connection': 'close',
    }
base_link = 'https://productcategory.net/'

def get_links_and_names(base_link, cur_headers):
    response = requests.get(base_link, headers=cur_headers)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    content = soup.find_all('a')
    links = []
    names = []
    for item in content:
        if 'finder' in str(item):
            raw_link, raw_name, _ = str(item).split('>')
            link = raw_link.split('=')[1][1:-1]
            name = clean_name(raw_name.split('<')[0])
            links.append(link)
            names.append(name)
    return links, names

#all_links, all_names = get_links_and_names(base_link=base_link, cur_headers=cur_headers)

#cur_link = all_links[0]

txt_path = 'TaxoGlimpse/LLM-taxonomy/shopping/data/google-US.txt'

def replace_and(string):
    return string.replace('&', 'and')

def replace_blank(string):
    return string.replace(' ','-')

def clean_each_level(current_tax_path):
    out_link = 'https://productcategory.net/finder/'
    cleaned_tax_path = []
    for item in current_tax_path:
        cleaned_tax_path.append(replace_blank(replace_and(item.strip())).lower())
    for item in cleaned_tax_path:
        out_link += item
        out_link += '/'
    return out_link

def compose_leaf_links_from_txt(input_txt):
    cleaned_links = []
    with open(input_txt, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for idx, line in enumerate(content):
            if idx == 0:
                continue
            else:
                current_tax_path = line.strip().split('>')
                if len(current_tax_path) == 1:
                    continue
                else:
                    cleaned_links.append(clean_each_level(current_tax_path))
    return cleaned_links

def leaf_page_to_product_page(cur_link, cur_headers):
    response = requests.get(cur_link, headers=cur_headers)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    target_text = 'View on Google Shopping'
    content = soup.find_all('a')
    for item in content:
        if target_text in str(item):
            raw_link = str(item).split('style')[0]
            link = clean_name(raw_link.strip().split('href=')[1][1:-1])
            response = requests.get(link, headers=cur_headers)
            html_content = response.content
            soup = BeautifulSoup(html_content, 'html.parser')
            content = soup.find_all('h3')
            cleaned_product_names = []
            for item in content:
                raw_name = str(item).split('</')[0]
                name = raw_name.split('>')[1]
                cleaned_product_names.append(name)
            return cleaned_product_names

def get_failed_links():
    dict_base_path = 'TaxoGlimpse/LLM-taxonomy/shopping/data/google_crawled/'

    failed_dict_keys = []
    for pickled_file_name in os.listdir(dict_base_path):
        pickled_file_path = dict_base_path + pickled_file_name
        with open(pickled_file_path, 'rb') as file:
            pickled_dict = dict(pickle.load(file))
        for key, value in pickled_dict.items():
            if not value:
                failed_dict_keys.append(key)
    return failed_dict_keys

def crawl_products(txt_path, cur_headers):
    cleaned_links = compose_leaf_links_from_txt(txt_path)
    cleaned_product_names_dict = {}
    failed_links = get_failed_links()
    failed_times = 0
    
    for cur_link in tqdm(cleaned_links):
        if cur_link in failed_links:
            sleep_time = random.random()*5
            time.sleep(sleep_time)
            ua = UserAgent()
            cur_ua = ua.random
            cur_headers = {'content-type': 'text/html; charset=UTF-8',
            'user-agent': cur_ua,
            'Referer': cur_link,
            'Connection': 'close',
            }
            cleaned_product_names = leaf_page_to_product_page(cur_link, cur_headers)
            cleaned_product_names_dict[cur_link] = cleaned_product_names
            if not cleaned_product_names:
                failed_times += 1
            if failed_times > 10:
                break
    return cleaned_product_names_dict

count = 13
product_dict = crawl_products(txt_path, cur_headers)
with open('TaxoGlimpse/LLM-taxonomy/shopping/data/crawled/product_dict'+str(count)+'.pkl', 'wb') as file:
    # 使用pickle的dump函数将字典序列化并存储到文件
    pickle.dump(product_dict, file)