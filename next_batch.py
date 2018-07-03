import os
import tensorflow as tf
import pandas as pd
import numpy as np
import train_test_split
import vec_to_array
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

table = pd.read_csv("final_table.csv")
X = table.loc[:,['vec','review_vec']]
y = table.loc[:,['label_vec']]
# split
X_train, X_test, y_train, y_test = train_test_split.stratify_split(X,y)

X_train, X_vali, y_train, y_vali = train_test_split.stratify_split(X_train, y_train)
# print(len(X_train), len(y_train))
# print(len(X_test), len(y_test))
# print(len(X_vali), len(y_vali))


# train
X_img_train = X_train.loc[:,['vec']]
X_review_train = X_train.loc[:,['review_vec']]
# vali
X_img_vali = X_vali.loc[:,['vec']]
X_review_vali = X_vali.loc[:,['review_vec']]
# test
X_img_test = X_test.loc[:,['vec']]
X_review_test = X_test.loc[:,['review_vec']]
# train_to_list
X_image_train_list = X_img_train['vec'].tolist()
X_review_train_list = X_review_train['review_vec'].tolist()
y_train_list = y_train['label_vec'].tolist()
# vali_to_list
X_image_vali_list = X_img_vali['vec'].tolist()
X_review_vali_list = X_review_vali['review_vec'].tolist()
y_vali_list = y_vali['label_vec'].tolist()
# test_to_list
X_image_test_list = X_img_test['vec'].tolist()
X_review_test_list = X_review_test['review_vec'].tolist()
y_test_list = y_test['label_vec'].tolist()



# train_to_array
X_image_train_array = vec_to_array.to_vec(X_image_train_list)
X_review_train_array= vec_to_array.to_vec(X_review_train_list)
y_train_array = vec_to_array.to_vec(y_train_list)
# vali_to_array
X_image_vali_array = vec_to_array.to_vec(X_image_vali_list)
X_review_vali_array= vec_to_array.to_vec(X_review_vali_list)
y_vali_array = vec_to_array.to_vec(y_vali_list)
# test_to_array
X_image_test_array = vec_to_array.to_vec(X_image_test_list)
X_review_test_array = vec_to_array.to_vec(X_review_test_list)
y_test_array = vec_to_array.to_vec(y_test_list)

def useful_data():
    train_x = X_image_train_array
    train_rx = X_review_train_array
    train_y = y_train_array
    vali_x = X_image_vali_array
    vali_rx = X_review_vali_array
    vali_y = y_vali_array
    test_x = X_image_test_array
    test_rx = X_review_test_array
    test_y = y_test_array
    return train_x, train_rx, train_y, vali_x, vali_rx, vali_y, test_x, test_rx, test_y

train_x, train_rx, train_y, vali_x, vali_rx, vali_y, test_x, test_rx, test_y = useful_data()


def batch(i,size,var):
    mod_start = (i*size) % len(var)
    mod_end = mod_start + size

    if mod_end > len(var):
        # batch = var[mod_start:] + var[:(mod_end - len(var))]
        batch = np.append(var[mod_start:], var[:(mod_end - len(var))], 0)
    else:
        batch = var[mod_start: mod_end]
    return batch


def next_batch(i,size):
    x = batch(i,size,train_x)
    y = batch(i,size,train_y)
    return x,y





