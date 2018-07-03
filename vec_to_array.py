import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import train_test_split


# data preparation
table = pd.read_csv("final_table.csv")
X = table.loc[:,['vec','review_vec']]
y = table.loc[:,['label_vec']]
X_train, X_test, y_train, y_test = train_test_split.stratify_split(X,y)
X_img_train = X_train.loc[:,['vec']]
X_review_train = X_train.loc[:,['review_vec']]
X_img_test = X_test.loc[:,['vec']]
X_review_test = X_test.loc[:,['review_vec']]


x_image_train_list = X_img_train['vec'].tolist()
x_review_train_list = X_review_train['review_vec'].tolist()
y_train_list = y_train['label_vec']


def to_vec(ori_list):
    new_dic_list = []
    for list in ori_list:
        list_1 = list.split('[')
        list_2 = list_1[1].split(']')
        list_3 = list_2[0].split(', ')
        new_list = []
        for item in list_3:
            new_list.append(float(item))
        # print(new_list)
        new_array = np.asarray(new_list)
        # print(new_array)
        new_dic_list.append(new_array)
    new_dic_array = np.asarray(new_dic_list)
    return new_dic_array



# print(x_image_train_array[:5])
# print(x_review_train_list[:5])
# print(y_train_array[:5])
