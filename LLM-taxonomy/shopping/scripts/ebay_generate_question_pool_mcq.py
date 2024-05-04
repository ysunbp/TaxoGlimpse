from bs4 import BeautifulSoup
import csv
import random
from tqdm import tqdm


# 读取 HTML 文件
with open('eBay.html', 'r') as file:
    html_content = file.read()

# 创建 BeautifulSoup 对象
soup = BeautifulSoup(html_content, 'html.parser')

# 找到所有的 <a> 标签
links = soup.find_all('a')

child2parent_dict_1 = {}
parent2child_dict_1 = {}

child2parent_dict_2 = {}
parent2child_dict_2 = {}

root_set = []

level_1 = []

level_2 = []
# 打印筛选结果
for link in links:
    if "top-cat" in str(link):
        #print(link) ## 获取首层
        h2_tag = link.find('h2')
        if h2_tag:
            text = h2_tag.text
            root_set.append(text.strip())
            cur_parent = text.strip()
    elif "cat-img-wrapper all-cats-lazy-load" in str(link):
        #print(link) ## 获取第二层 
        if 'Top Brands' in str(link) or 'Top Stores' in str(link) or 'Popular Topics' in str(link):
            continue
        img_tag = link.find('img')
        if img_tag:
            alt_value = img_tag.get('alt')
            level_1_node = alt_value.strip()
            if level_1_node in child2parent_dict_1.keys():
                child2parent_dict_1[level_1_node].append(cur_parent)
            level_1.append(level_1_node)
            child2parent_dict_1[level_1_node] = [cur_parent]
            if not cur_parent in parent2child_dict_1.keys():
                parent2child_dict_1[cur_parent] = [level_1_node]
            else:
                parent2child_dict_1[cur_parent].append(level_1_node)
            cur_sub_parent = level_1_node
    elif "title" in str(link) and "href=\"https://www.ebay.com/b/" in str(link):
        #print(link) ## 获取第三层
        if 'Top Brands' in str(link) or 'Top Stores' in str(link) or 'Popular Topics' in str(link):
            continue
        title_value = link.get('title')
        if title_value:
            level_2_node = title_value.strip()
            if level_2_node in child2parent_dict_2.keys():
                #print(level_2_node, '|', cur_sub_parent)
                child2parent_dict_2[level_2_node].append(cur_sub_parent)
            level_2.append(level_2_node)
            child2parent_dict_2[level_2_node] = [cur_sub_parent]
            if not cur_sub_parent in parent2child_dict_2.keys():
                parent2child_dict_2[cur_sub_parent] = [level_2_node]
            else:
                parent2child_dict_2[cur_sub_parent].append(level_2_node)

print(len(list(set(level_1))))

print(len(list(set(level_2))))

level_nodes = [root_set, level_1, level_2]

def get_cur_level_uncles(cur_level_nodes, level_nodes, level):
    uncle_dict = {}
    if level == 1:
        for node in cur_level_nodes:
            uncle_dict[node] = list(set(level_nodes[0])-set(child2parent_dict_1[node]))
        return uncle_dict
    else:
        for node in cur_level_nodes:
            cur_parent_node = child2parent_dict_2[node][0]
            cur_grand_node = child2parent_dict_1[cur_parent_node][0]
            cur_uncles = parent2child_dict_1[cur_grand_node]
            if len(cur_uncles) > len(child2parent_dict_2[node]):
                uncle_dict[node] = list(set(cur_uncles)-set(child2parent_dict_2[node]))
            else:
                uncle_dict[node] = list(set(level_nodes[level-1])-set(child2parent_dict_2[node]))
        return uncle_dict

def sample_question_pool(cur_level_nodes, level_nodes, nodes_uncles_dict, sample_size, cur_level, out_path, question_mode=0):
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        if question_mode == 0: # MCQ hard
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            for node in tqdm(sampled_nodes):
                if len(nodes_uncles_dict[node]) < 3:
                    selected_mcqs = nodes_uncles_dict[node] + random.sample(list(set(level_nodes[cur_level])-set([node])), int(3-len(nodes_uncles_dict[node])))
                else:
                    selected_mcqs = random.sample(nodes_uncles_dict[node], 3)
                if cur_level == 1:
                    cur_template = ('shopping-ebay', child2parent_dict_1[node][0], node, cur_level, 'mcq-negative-hard', selected_mcqs[0], selected_mcqs[1], selected_mcqs[2])
                else:
                    cur_template = ('shopping-ebay', child2parent_dict_2[node][0], node, cur_level, 'mcq-negative-hard', selected_mcqs[0], selected_mcqs[1], selected_mcqs[2])
                
                csv_writer.writerow(cur_template)
            print('size of the MCQ set', len(sampled_nodes), cur_level)

        elif question_mode == 1: # MCQ easy
            if len(cur_level_nodes) < sample_size:
                sampled_nodes = cur_level_nodes
            else:
                sampled_nodes = random.sample(cur_level_nodes, sample_size)
            for node in tqdm(sampled_nodes):
                
                if cur_level == 1:
                    if len(list(set(level_nodes[cur_level-1])-set(child2parent_dict_1[node]))) < 3:
                        selected_mcqs = list(set(level_nodes[cur_level-1])-set(child2parent_dict_1[node])) + random.sample(list(set(level_nodes[cur_level])-set([node])), int(3-len(nodes_uncles_dict[node])))
                    else:
                        selected_mcqs = random.sample(list(set(level_nodes[cur_level-1])-set(child2parent_dict_1[node])), 3)
                    cur_template = ('shopping-ebay', child2parent_dict_1[node][0], node, cur_level, 'mcq-negative-easy', selected_mcqs[0], selected_mcqs[1], selected_mcqs[2])
                else:
                    if len(list(set(level_nodes[cur_level-1])-set(child2parent_dict_2[node]))) < 3:
                        selected_mcqs = list(set(level_nodes[cur_level-1])-set(child2parent_dict_2[node])) + random.sample(list(set(level_nodes[cur_level])-set([node])), int(3-len(nodes_uncles_dict[node])))
                    else:
                        selected_mcqs = random.sample(list(set(level_nodes[cur_level-1])-set(child2parent_dict_2[node])), 3)
                    
                    cur_template = ('shopping-ebay', child2parent_dict_2[node][0], node, cur_level, 'mcq-negative-easy', selected_mcqs[0], selected_mcqs[1], selected_mcqs[2])
                
                csv_writer.writerow(cur_template)
            print('size of the MCQ set', len(sampled_nodes), cur_level)
        


num_question_samples = [100000, 100000]

file_name = ['mcq_hard.csv', 'mcq_easy.csv']


for cur_question_mode in range(2):
    out_path = 'TaxoGlimpse/question_pools/shopping-ebay/mcq/question_pool_full_' + file_name[cur_question_mode]
    with open(out_path, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['domain', 'parent', 'child', 'level', 'type', 'mcq-parent1', 'mcq-parent2', 'mcq-parent3'])  # 写入表头
    
    for cur_level in range(1, len(num_question_samples)+1):
        nodes_uncles_dict = get_cur_level_uncles(level_nodes[cur_level], level_nodes, cur_level) # level 1是从level 1 到root
        sample_question_pool(level_nodes[cur_level], level_nodes, nodes_uncles_dict, sample_size=num_question_samples[cur_level-1], cur_level=cur_level, out_path=out_path, question_mode=cur_question_mode)
