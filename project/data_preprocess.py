import pandas as pd
import numpy as np
import random

def data_patition():
    raw_data = pd.read_csv('../data/data.csv')
    label_map = {'sleep': 1, "awake": 0}

    sleep_data = raw_data.loc[raw_data['睡眠状态'] == 1, :]
    sleep_data = sleep_data.reset_index(drop=True)
    awake_data = raw_data.loc[raw_data['睡眠状态'] == 0, :]
    awake_data = awake_data.reset_index(drop=True)
    # print(len(sleep_data))
    # print(len(awake_data))

    # 17
    index1 = list(range(len(sleep_data)))
    index2 = list(range(len(awake_data)))

    random.shuffle(index1)
    random.shuffle(index2)
    test_index1 = index1[:17] #sleep
    train_index1 = index1[17:]# awake

    test_index2 = index2[:17]
    train_index2 = index2[17:]

    test_data = pd.DataFrame(columns= raw_data.columns)
    train_data = pd.DataFrame(columns= raw_data.columns)
    for i in range(len(raw_data)):
        if i in test_index1:
            test_data.loc[test_data.shape[0]] = sleep_data.loc[i]
        elif i in train_index1:
            train_data.loc[train_data.shape[0]] = sleep_data.loc[i]

        if i in test_index2:
            test_data.loc[test_data.shape[0]] = awake_data.loc[i]
        elif i in train_index2:
            train_data.loc[train_data.shape[0]] = awake_data.loc[i]


    train_data.to_csv('../data/train/train.csv', index = False, encoding = 'gbk')
    test_data.to_csv('../data/test/test.csv', index = False, encoding = 'gbk')



    # test_data =
# random.random()
# print('ok')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_patition()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
