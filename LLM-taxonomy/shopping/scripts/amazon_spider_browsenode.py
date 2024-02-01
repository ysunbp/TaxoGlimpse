from bs4 import BeautifulSoup
import requests
from tqdm import trange
import os
import random
import time
import pickle
from fake_useragent import UserAgent

ua = UserAgent()

amazon_base_page = 'https://www.browsenodes.com/amazon.com'
amazon_base_domain = 'https://www.browsenodes.com'

node_names = ['Appliances', 'Arts, Crafts and Sewing', 'Automotive', 'Baby', 'Beauty', 'Books', 'Collectibles and Fine Arts', 'Electronics', 'Clothing, Shoes and Jewelry', 'Clothing, Shoes and Jewelry - Baby', 'Clothing, Shoes and Jewelry - Boys', 'Clothing, Shoes and Jewelry - Girls', 'Clothing, Shoes and Jewelry - Men', 'Clothing, Shoes and Jewelry - Women', 'Gift Cards', 'Grocery and Gourmet Food', 'Handmade', 'Health and Personal Care', 'Home and Kitchen', 'Industrial and Scientific', 'Kindle Store', 'Patio, Lawn and Garden', 'Luggage and Travel Gear', 'Magazine Subscriptions', 'Apps and Games', 'Movies and TV', 'Digital Music', 'CDs and Vinyl', 'Musical Instruments', 'Office Products', 'Computers', 'Pet Supplies', 'Software', 'Sports and Outdoors', 'Tools and Home Improvement', 'Toys and Games', 'Amazon Instant Video', 'Vehicles', 'Video Games', 'Wine', 'Cell Phones and Accessories']
node_links = ['https://www.browsenodes.com/amazon.com/browseNodeLookup/2619526011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2617942011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/15690151.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/165797011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/11055981.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/1000.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/4991426011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/493964.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7141124011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147444011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147443011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147442011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147441011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/7147440011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2864120011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/16310211.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/11260433011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/3760931.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/1063498.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/16310161.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/133141011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/3238155011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/9479199011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/599872.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2350150011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2625374011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/624868011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/301668.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/11965861.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/1084128.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/541966.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2619534011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/409488.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/3375301.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/468240.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/165795011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2858778011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/10677470011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/11846801.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2983386011.html', 'https://www.browsenodes.com/amazon.com/browseNodeLookup/2335753011.html']


class TreeNode:
    def __init__(self, headers, amazon_base_domain='https://www.browsenodes.com', page='https://www.browsenodes.com/amazon.com', name='root', is_leaf=False):
        self.name = name
        self.children = []
        self.is_leaf = is_leaf
        self.page = page
        self.base_domain = amazon_base_domain
        self.headers = headers

        response = requests.get(self.page, headers=headers)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        raw_links = soup.find_all('tr')

        if 'leaf' in str(raw_links):
            self.is_leaf = True
        else:
            self.add_children(raw_links)
        
        print('cur node', self.name)
        print('cur children')
        for child in self.children:
            print(child.name)
        print('===================')

    def add_children(self, raw_links):
        for item in raw_links:
            identified_node = item.find_all('td')
            if len(identified_node) == 0:
                continue
            splitted = str(identified_node).split('>')
            name = splitted[1]
            for splitted_item in splitted:
                if 'read-more' in str(splitted_item):
                    link = str(splitted_item)
            name = name.split('<')[0].strip()
            name = self.clean_name(name)
            link = self.base_domain+link.split('href=')[1][1:-1]
            cur_node = TreeNode(page=link, name=name, headers=self.headers)
            self.children.append(cur_node)

    def clean_name(self, string):
        return string.replace('&amp;', 'and')



def find_cur_level(page, amazon_base_domain):
    if 'leaf' in str(page):
        print('reach leaf')
        return None
    else:
        cur_names = []
        cur_links = []
        for item in page:
            identified_node = item.find_all('td')
            if len(identified_node) == 0:
                continue
            splitted = str(identified_node).split('>')
            name = splitted[1]
            for splitted_item in splitted:
                if 'read-more' in str(splitted_item):
                    link = str(splitted_item)
            name = name.split('<')[0].strip()
            name = clean_name(name)
            link = amazon_base_domain+link.split('href=')[1][1:-1]
            cur_names.append(name)
            cur_links.append(link)
        return (cur_names, cur_links)

for idx in trange( len(node_names)):
    cur_ua = ua.random
    cur_name = node_names[idx]
    cur_link = node_links[idx]
    cur_headers = {'content-type': 'text/html; charset=UTF-8',
    'user-agent': cur_ua,
    'Referer': 'https://www.google.com/',
    'Connection': 'close'
    }
    cur_tree = TreeNode(page=cur_link, name=cur_name, headers=cur_headers)
    with open('TaxoGlimpse/LLM-taxonomy/shopping/data/browsenodes/'+str(idx)+'.pkl', 'wb') as pickle_file:
        pickle.dump(cur_tree, pickle_file)


