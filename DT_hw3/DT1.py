from sklearn.datasets import load_breast_cancer
from sklearn import tree, metrics
import pandas as pd
import numpy as np
# import pydotplus
# from IPython.display import Image

import sys


# import os
# import csv
# import shutil
# import warnings
# from collections import namedtuple
# from os import environ, listdir, makedirs
# from os.path import dirname, exists, expanduser, isdir, join, splitext
# import hashlib

from sklearn.utils import check_random_state



if __name__ == '__main__':
    # os.environ["PATH"] += os.pathsep + os.pathsep.join(["C:\\Users\\kfir\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\graphviz","C:\\Program Files (x86)\\Graphviz2.38\\bin"])
    # print("C:\\Users\\kfir\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\graphviz\n")
    # print("PATH: {}\n".format(os.getenv('Path')))
    # if "C:\\Users\\kfir\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\graphviz" in str(os.getenv('PATH')):
    #     print("true")
    # else:
    #     print("false")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_np = train_df.to_numpy()
    test_np = test_df.to_numpy()

    x_train, y_train = train_np[:, :-1], train_np[:, -1]
    x_test, y_test = test_np[:, :-1], test_np[:, -1]

    DT_ID3 = tree.DecisionTreeClassifier(criterion="entropy")
    DT_ID3 = DT_ID3.fit(x_train, y_train)
    y_pred = DT_ID3.predict(x_test)
    print("Confusion matrix:\n{}\n".format(metrics.confusion_matrix(y_test, y_pred)))
    # open csv file for tree result
    # f = open("experiments.csv", 'w')
    # for i in [3, 9, 27]:
    #     DT_ID3 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=i)
    #     DT_ID3 = DT_ID3.fit(x_train, y_train)
    #     y_pred = DT_ID3.predict(x_test)
    #     f.write("{}, {}\n".format(i, metrics.accuracy_score(y_test, y_pred)))
    #     print("Confusion matrix for {}:\n{}\n".format(i, metrics.confusion_matrix(y_test, y_pred)))
    # tree.plot_tree(DT_ID3)
    # tree_data = tree.export_graphviz(DT_ID3)
    # Draw graph
    # graph = pydotplus.graph_from_dot_data(tree_data)
    # Show graph
    # Image(graph.create_png())
    # graph.write_png("DT.png")
    # f.close()
