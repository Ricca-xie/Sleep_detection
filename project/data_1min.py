import pandas as pd
import random
raw_data = pd.read_csv('../data/data_1min/res_20220429v2-1.csv')
# print(data)
sleep_data = raw_data.loc[raw_data['sleep_value'] == 1, :]
sleep_data = sleep_data.reset_index(drop=True)
awake_data = raw_data.loc[raw_data['sleep_value'] == 0, :]
awake_data = awake_data.reset_index(drop=True)

index1 = list(range(len(sleep_data)))
index2 = list(range(len(awake_data)))
random.shuffle(index1)
random.shuffle(index2)

all_len = len(raw_data)
train_len = int(all_len*0.8)
test_len = all_len - train_len


test_index1 = index1[:test_len//2] #sleep
train_index1 = index1[test_len//2:]# awake

test_index2 = index2[:test_len//2]
train_index2 = index2[test_len//2:]
print(all_len)
test_data = pd.DataFrame(columns= raw_data.columns)
train_data = pd.DataFrame(columns= raw_data.columns)
for i in range(all_len):
    print(i)
    if i in test_index1:
        test_data.loc[test_data.shape[0]] = sleep_data.loc[i]
    elif i in train_index1:
        train_data.loc[train_data.shape[0]] = sleep_data.loc[i]

    if i in test_index2:
        test_data.loc[test_data.shape[0]] = awake_data.loc[i]
    elif i in train_index2:
        train_data.loc[train_data.shape[0]] = awake_data.loc[i]


train_data.to_csv('../data/data_1min/train/train.csv', index = False, encoding = 'gbk')
test_data.to_csv('../data/data_1min/test/test.csv', index = False, encoding = 'gbk')
