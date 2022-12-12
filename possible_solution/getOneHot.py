import pandas as pd
import numpy as np

# 'train' or 'val'
def prepare_df(train_val):
    df1 = pd.read_csv('./data/train.csv' if train_val=='train' else './data/val_seen.csv')
    df2 = pd.read_csv('./data/users.csv')
    merged = pd.merge(df1,df2,how='left',on=['user_id'])
    df3 = merged[['user_id', 'course_id', 'interests']].dropna()
    return df3

def get_data(train_val):
    subgroup_csv = pd.read_csv('./data/subgroups.csv')
    courses_csv = pd.read_csv('./data/courses.csv').fillna('')
    df = prepare_df(train_val)
    bought_course_onehot = np.zeros((len(df), len(subgroup_csv)))

    course_err = []
    for i, course_lst in enumerate(df['course_id']):
        for course in course_lst.split(' '):
            course_sub = courses_csv.loc[courses_csv['course_id'] == course, 'sub_groups'].iloc[0]
            if(course_sub == ''):
                continue
            for sub in course_sub.split(','):
                try:
                    bought_course_onehot[i][subgroup_csv.loc[subgroup_csv['subgroup_name'] == sub, 'subgroup_id'].iloc[0] - 1] = 1
                except IndexError:
                    if(sub not in course_err):
                        course_err.append(sub)

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

    data = {'user_id': df['user_id'].to_numpy(), 'input': self_interest_onehot, 'label': bought_course_onehot}
    return data