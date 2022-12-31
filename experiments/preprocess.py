import pandas as pd
import json
import torch
import os

df = pd.read_csv('./data/users.csv', encoding='utf-8').fillna('')

def create_interests_json():
    interest_list = []
    for i in df['interests'].tolist():
        interest_list.extend(i.split(','))
    interest_list = list(set(interest_list))[1:]
    interest_list.sort()

    interest_dict = {'':-1}
    s = '人文'
    grp_ct = 1
    sub_ct = 1
    for i in interest_list:
        grp, _ = i.split('_')
        if(grp == s):
            interest_dict[i] = grp_ct * 100 + sub_ct
            sub_ct += 1
        else:
            grp_ct += 1
            interest_dict[i] = grp_ct * 100 + sub_ct
            s = grp
    
    with open ('./data/interests.json', 'w', encoding='utf-8') as f:
        json.dump(interest_dict, f, indent=4, ensure_ascii=False)

if(not os.path.exists('./data/interests.json')):
    create_interests_json()
with open('./data/interests.json', 'r', encoding='utf-8') as f:
    interests = json.load(f)
device = 'cuda' if torch.cuda.is_available() else 'cpu'   

# turn user interest in user.csv to tensor with shape(128), fill with -1
def get_user_interest():
    user_interests = df['interests'].to_list()
    interest_tensor = []
    for u_interests in user_interests:
        t = [-1] * 128
        u_interest = u_interests.split(',')
        for i in range(len(u_interest)):
            t[i] = interests[u_interest[i]]
        interest_tensor.append(t)
    interest_tensor = torch.tensor(interest_tensor).to(device)
    return interest_tensor

# turn user bought courses in train.csv to interests with shape(64), fill with -1
def get_train_data_labels():
    train_data = pd.read_csv('./data/train.csv')
    courses = pd.read_csv('./data/courses.csv', encoding='utf-8').fillna('')
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
        bought_courses_tensor.append(bought_courses + [-1] * (64 - len(bought_courses)))
    bought_courses_tensor = torch.tensor(bought_courses_tensor).to(device)
    return bought_courses_tensor

def create_user_list_json():
    train_data = pd.read_csv('./data/train.csv')
    user_list = train_data['user_id'].to_list()
    user_dict = {}
    for i, users in enumerate(user_list):
        user_dict[users] = str(i)
    with open ('./data/user_list.json', 'w', encoding='utf-8') as f:
        json.dump(user_dict, f, indent=4, ensure_ascii=False)
