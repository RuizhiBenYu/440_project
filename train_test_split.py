import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# table = pd.read_csv("final_table.csv")
# table = table[:200]
# print(table.head(5))


# X = table.loc[:,['image_vec','review_vec']]
# y = table['label_vec']
# print(X)
# print(y)


def stratify_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test
