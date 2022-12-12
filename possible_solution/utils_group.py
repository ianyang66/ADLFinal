import random
import pandas as pd
import json
import torch
import math

GROUPS = {  '00_空值':[0], '01_人文':[62, 71, 85], '02_手作':[9, 17, 18, 24, 27, 31, 46, 89], \
            '03_投資理財':[64, 69, 70, 76], \
            '04_攝影':[19, 40, 53, 56, 73, 86, 90], \
            '05_生活品味':[1, 2, 13, 14, 15, 16, 29, 55, 79, 82, 84], \
            '06_程式':[32, 33, 34, 41, 42, 43, 44, 52, 57, 58, 63, 68, 74, 75, 80, 81], \
            '07_職場技能':[7, 35, 37,50, 51, 54, 59, 61, 78, 83, ], \
            '08_藝術':[4, 5, 12, 20, 22, 23, 36, 60], '09_行銷':[65, 66, 72, 87, 91], \
            '10_設計':[3, 6, 25, 38, 39, 67, 77], '11_語言':[8, 28, 45, 47, 48, 88], '12_音樂':[10, 11, 21, 26, 30, 49]}
RELATION_MATRIX = [
    [4, 3, 1, 1, 3, 1, 1, 3, 2, 3, 3, 3],
    [3, 4, 1, 1, 3, 1, 1, 3, 1, 3, 1, 2],
    [1, 1, 4, 1, 2, 2, 3, 1, 1, 1, 1, 1],
    [1, 1, 1, 4, 3, 1, 2, 3, 3, 3, 1, 1],
    [3, 3, 2, 3, 4, 1, 1, 3, 2, 3, 1, 3],
    [1, 1, 2, 1, 1, 4, 3, 1, 1, 2, 1, 1],
    [1, 1, 3, 2, 1, 3, 4, 1, 3, 2, 2, 1],
    [3, 3, 1, 3, 3, 1, 1, 4, 3, 3, 1, 3],
    [2, 1, 1, 3, 2, 1, 3, 3, 4, 3, 1, 1],
    [3, 3, 1, 3, 3, 2, 2, 3, 3, 4, 1, 2],
    [3, 1, 1, 1, 1, 1, 2, 1, 1, 1, 4, 2],
    [3, 2, 1, 1, 3, 1, 1, 3, 1, 2, 2, 4]
    ]

random.seed(55)
device = 'cuda' if torch.cuda.is_available() else 'cpu'   


train_data = pd.read_csv('data/train_group.csv').fillna('0')
val_data = pd.read_csv('data/val_seen_group.csv').fillna('0')
with open('data/user_list.json', 'r', encoding='utf-8') as f:
    user_list = json.load(f)

group_dict = {}
for i in GROUPS.keys():
    for j in GROUPS[i]:
        group_dict[j] = int(i.split('_')[0])
print(group_dict)
def generate_txt(data):
    subgroups = data['subgroup'].to_list()
    user_ids = data['user_id'].to_list()
    s = ''
    def get_relation_score(samples, user_sub):
        group = group_dict[samples]
        m = 0
        for user_sub_to_group in user_sub:
            r = group_dict[int(user_sub_to_group)]
            m += RELATION_MATRIX[r-1][group-1]
        return str(round(m / len(user_sub)))


    for id, user_sub in zip(user_ids, subgroups):
        #series = random.randint(6, 15)
        user_sub = user_sub.split(' ')
        for sub in user_sub:
            if(sub == '0'):
                s += (user_list[id] + '\t' + '8' + '\t1\t98765432\n')
            else:
                s += (user_list[id] + '\t' + str(sub) + '\t1\t98765432\n')
        #m = min(series - len(user_sub), 3 * len(user_sub))
        #x = [int(i) for i in [str(x) for x in range(92)] if i not in user_sub]
        #sample_choice = random.choices(x, k = m)
        #for samples in sample_choice:
            #s += (str(user_list[id]) + '\t' + str(samples) + '\t' + get_relation_score(samples, user_sub) + '\t98765432\n')
    return s

def generate_test_txt():
    test_data = pd.read_csv('./data/test_seen.csv', encoding='utf-8')
    user_to_idx = ''
    for user in test_data['user_id'].to_list():
        user_to_idx += user_list[user] + '\n'
    with open('./possible_solution/hahow-100k-test.txt', 'w', encoding='utf-8') as f:
        f.write(user_to_idx)

# with open('./possible_solution/hahow-100k-train-group.txt', 'w', encoding='utf-8') as f:
#     f.write(generate_txt(train_data))
# with open('./possible_solution/hahow-100k-val-seen-group.txt', 'w', encoding='utf-8') as f:
#     f.write(generate_txt(val_data))
   

def generate_submit():
    subgroup_result = pd.read_csv('./possible_solution/pre_subgroup.csv')['subgroup'].to_list()
    group = pd.read_csv('./data/test_seen_group.csv')['user_id'].to_list()

    s = 'user_id,subgroup\n'
    for user in group:
        s += user +',' + (' '.join([str(int(x) + 1) for x in subgroup_result[int(user_list[user])].split(' ')])) + '\n'
    with open('./possible_solution/submit_seen.csv', 'w') as f:
        f.write(s)

generate_submit()
