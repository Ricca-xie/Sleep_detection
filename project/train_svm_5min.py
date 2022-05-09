import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import logging
import os

def logger_config(file_name=None):
    logger = logging.getLogger()
    logger.setLevel("INFO")
    basic_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(basic_format, date_format)
    console_handler = logging.StreamHandler()  # output to console
    console_handler.setFormatter(formatter)
    console_handler.setLevel("INFO")
    if file_name:
        file_handler = logging.FileHandler(file_name, mode="w")  # output to file
        file_handler.setFormatter(formatter)
        file_handler.setLevel("INFO")
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# logger = logger_config(os.path.join("logs", f"{args.backbone}_{args.log_file}"))
logger = logger_config(file_name='./svmlog5')
raw_data = pd.read_csv('../data/data_1min/res_20220429v2-1.csv')
raw_data = raw_data.fillna(value=0)
# print(data)
sleep_data = raw_data.loc[raw_data['sleep_value'] == 1, :]
sleep_data = sleep_data.reset_index(drop=True)
awake_data = raw_data.loc[raw_data['sleep_value'] == 0, :]
awake_data = awake_data.reset_index(drop=True)

all_len = len(raw_data)
# train_len = int(all_len*0.8)
# test_len = all_len - train_len

X = []
Y = []
feature = raw_data.loc[:,['heart_value','breath_value','snore_value','pulsesum_value','pressuresmall_value','pressurebig_value','pressureloinsmall_value','pressureloinbig_value']]
label = raw_data.loc[:,'sleep_value']

for i in range(0,all_len,5):
    # print(i)
    logger.info(i)
    x = []
    y = []
    class0 = 0
    class1 = 0
    for j in range(i, i+5):
        if j == all_len:
            break
        x.extend(list(feature.iloc[j, :]))
        if label[j] == 0 :class0 += 1
        if label[j] == 1: class1 += 1
    if class0 > class1:
        label_i = 0
    else:
        label_i = 1
    if len(x)<40:
        continue
    X.append(x)
    Y.append([label_i])
    # if i == 100:
    #     break

X = np.mat(X)
Y = np.mat(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state= 1, test_size=0.2)

c_value = [0.1]
for c_value in c_value:
    svm_model = svm.SVC(C = c_value)
    svm_model.fit(X_train, Y_train)
    Y_pred = svm_model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    logger.info("C = {}, 准确率： {}".format(c_value, acc))

