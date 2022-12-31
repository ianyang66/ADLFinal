import pandas as pd
import numpy as np
import json



# 'train' or 'val'
def prepare_train_val_df(train_val):
    df1 = pd.read_csv('./data/train.csv' if train_val=='train' else './data/val_seen.csv')
    df2 = pd.read_csv('./data/users.csv')
    merged = pd.merge(df1,df2,how='left',on=['user_id'])
    df3 = merged[['user_id', 'course_id', 'interests']].dropna()
    return df3

def prepare_test_df(seen_unseen):
    df1 = pd.read_csv('./data/test_' + seen_unseen + '.csv')
    df2 = pd.read_csv('./data/users.csv')
    merged = pd.merge(df1,df2,how='left',on=['user_id'])
    df3 = merged[['user_id', 'interests']].fillna('_')
    return df3
    
def get_data(train_val_test, seen_unseen='seen'):
    subgroup_csv = pd.read_csv('./data/subgroups.csv')
    if(train_val_test != 'test'):
        df = prepare_train_val_df(train_val_test)
        bought_course_onehot = np.zeros((len(df), len(subgroup_csv)))
        with open('./data/course_subgroup_idx.json') as f:
            cs_idx_dict = json.load(f)
        for i, course_lst in enumerate(df['course_id']):
            for course in course_lst.split(' '):
                course_sub = cs_idx_dict[course]
                if(course_sub == []):
                    continue
                for sub in course_sub:
                    bought_course_onehot[i][sub] = 1
    else:
        df = prepare_test_df(seen_unseen)

    subgroup_dict = {}
    for i, sub_name in enumerate(subgroup_csv['subgroup_name']):
        subgroup_dict[sub_name] = i+1

    self_interest_onehot = np.zeros((len(df), len(subgroup_csv)))

    interest_err = []
    for i, interest_lst in enumerate(df['interests']):
        for interest in interest_lst.split(','):
            interest_sub = interest.split('_')[1]
            try:
                self_interest_onehot[i][subgroup_dict[interest_sub] - 1] = 1
            except KeyError:
                if(interest_sub not in interest_err):
                    interest_err.append(interest_sub)

    data = []
    if(train_val_test != 'test'):
        for id, input, label in zip(df['user_id'].to_numpy(), self_interest_onehot, bought_course_onehot):
            data.append({'user_id':id, 'input':input, 'label':label})
    else:
        for id, input in zip(df['user_id'].to_numpy(), self_interest_onehot):
            data.append({'user_id':id, 'input':input})
    return data