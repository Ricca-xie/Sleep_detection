import numpy as np
import pandas as pd
import os
import torch
import random

def Dataset(data):
    dataset = []

    for i in range(len(data)):
        label = data.loc[i, "睡眠状态"]
        feature = [eval(data.iloc[i, x]) for x in range(8)]
        feature = np.array(feature).T
        y = np.array(np.repeat(0, 2))
        y[label] = 1
        dataset.append((feature, y))

    return dataset



def data_loader(root_path, data_name, batch_size=12, type = 'spec-n', is_append=False,is_shuffle = True,is_single = False):

    data_path = os.path.join(root_path, data_name)
    data = pd.read_csv(data_path, encoding = 'gbk')

    dataset  = Dataset(data)
    # dataset = []
    # for X, Y in zip(data, label):
    #     dataset.append((X, Y))
    # print('finish creating dataset...')
    if is_append:
        print('is_training...')
        data_path2 = os.path.join(root_path, 'X_dev_'+type+'.myarray')
        label_path2 = os.path.join(root_path, 'Y_dev_'+type+'.myarray')
        # data2 = np.memmap(data_path2, dtype=np.float64, mode='r', shape=(13766, 3, 1, 300, 64))
        # label2 = np.memmap(label_path2, dtype=np.int32, mode='r', shape=(13766, 1251))
        data2 = np.load(data_path2)
        label2 = np.load(label_path2)
        for X, Y in zip(data2, label2):
            dataset.append((X, Y))

    if is_shuffle:
        random.shuffle(dataset)
    # print('finish shuffle...')
    # import pdb; pdb.set_trace()
    if is_single:
        for idx, value in enumerate(dataset):
            train_data, label = torch.FloatTensor(value[0]).unsqueeze(0), torch.LongTensor(value[1]).unsqueeze(0)
            yield (train_data, label)
    else:
        for idx, value in enumerate(dataset):
            if idx == 0:
                train_data, label = torch.FloatTensor(value[0]).unsqueeze(0), torch.LongTensor(value[1]).unsqueeze(0)
                # print(train_data.shape)
            elif idx % batch_size == 0:
                yield (train_data, label)
                train_data, label = torch.FloatTensor(value[0]).unsqueeze(0), torch.LongTensor(value[1]).unsqueeze(0)
            else:
                train_data = torch.cat((train_data, torch.FloatTensor(value[0]).unsqueeze(0)), 0)
                label = torch.cat((label, torch.LongTensor(value[1]).unsqueeze(0)), 0)

