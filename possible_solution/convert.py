import random
import pandas as pd
import json
import torch
import math

GROUPS = {'人文':0, '手作':1, '投資理財':2, '攝影':3, '生活品味':4, '程式':5, 
            '職場技能':6, '藝術':7, '行銷':8, '設計':9, '語言':10, '音樂':11}
RELATION_MATRIX = [
    [4, 3, 2, 3, 3, 1, 1, 3, 2, 3, 3, 3],
    [3, 4, 1, 1, 3, 1, 1, 3, 1, 3, 1, 2],
    [2, 1, 4, 1, 2, 2, 3, 1, 1, 1, 1, 1],
    [3, 1, 1, 4, 3, 1, 2, 3, 3, 3, 1, 1],
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


train_data = pd.read_csv('data/train.csv')
val_data = pd.read_csv('data/val_seen.csv')

def generate_txt(data):
    with open('data/interests.json', 'r', encoding='utf-8') as f:
        interests = json.load(f)
    courses = pd.read_csv('data/courses.csv', encoding='utf-8').fillna('')

    course_group = {}
    for (course_id, cell) in zip(courses['course_id'], courses['groups+subgroups']):
        interest_list = cell.split(',')
        t = []
        for i, int_ in enumerate(interest_list):
            t.append(interests[int_])
        course_group[course_id] = t

    bought_courses_list = []
    for user_id, courses in zip(data['user_id'], data['course_id']):
        bought_courses = []
        for user_bought in courses.split(' '):
            bought_courses.extend(course_group[user_bought])
        bought_courses = list(set(bought_courses))
        bought_courses_list.append([user_id, bought_courses]) 

    print(bought_courses_list[:10])
    interest_id = list(interests.values())
    interest_id.remove(0)


    def get_relation_score(sample, course_num):
        s = int(sample) // 100 - 1
        m = 0
        for c in course_num:
            c = c // 100 - 1
            if(s == c):
                return '4'
            else:
                m += RELATION_MATRIX[s][c]
        return str(math.floor(m / len(course_num)))

    s = ''
    for idx, (user_id, course_num) in enumerate(bought_courses_list):
        series = random.randint(6, 15)

        for course in course_num:
            s += (str(idx) + '\t' + str(course) + '\t5\t98765432\n')
        m = max(series - len(course_num), 5)
        sample_choice = random.choices(list(set(interest_id).difference(set(course_num))), k = m)
        for samples in sample_choice:
            s += (str(idx) + '\t' + str(samples) + '\t' + get_relation_score(samples, course_num) + '\t98765432\n')
    return s

with open('./possible_solution/hahow-100k-train.txt', 'w', encoding='utf-8') as f:
    f.write(generate_txt(train_data))
with open('./possible_solution/hahow-100k-val-seen.txt', 'w', encoding='utf-8') as f:
    f.write(generate_txt(val_data))
   