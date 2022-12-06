import random
import pandas as pd
import json
import torch

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

def get_train_data_labels():
    with open('data/interests.json', 'r', encoding='utf-8') as f:
        interests = json.load(f)
    train_data = pd.read_csv('data/train.csv')
    courses = pd.read_csv('data/courses.csv', encoding='utf-8').fillna('')
    course_group = {}
    for (course_id, cell) in zip(courses['course_id'], courses['groups+subgroups']):
        interest_list = cell.split(',')
        t = []
        for i, int_ in enumerate(interest_list):
            t.append(interests[int_])
        course_group[course_id] = t

    bought_courses_tensor = []
    for user_id, courses in zip(train_data['user_id'], train_data['course_id']):
        bought_courses = []
        for user_bought in courses.split(' '):
            bought_courses.extend(course_group[user_bought])
        bought_courses = list(set(bought_courses))
        bought_courses_tensor.append(bought_courses) # + [-1] * (64 - len(bought_courses)))
    #bought_courses_tensor = torch.tensor(bought_courses_tensor).to(device)
    return bought_courses_tensor

bought_courses = get_train_data_labels()
print()