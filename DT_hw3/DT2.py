from sklearn.datasets import load_breast_cancer
from sklearn import tree, metrics
import pandas as pd
import numpy as np
import pydotplus
from IPython.display import Image
import os

import sys

from sklearn.utils import check_random_state

if __name__ == '__main__':
    os.environ["PATH"] += os.pathsep + os.pathsep.join(["C:\\Users\\kfir\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\graphviz","C:\\Program Files (x86)\\Graphviz2.38\\bin"])
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    alpha = 4
    delat = 0.2
    train_data = pd.read_csv('train.csv', header=None, names=col_names)
    test_data = pd.read_csv('test.csv', header=None, names=col_names)
    train_data.head()
    test_data.head()
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    x_train = train_data[feature_cols]
    y_train = train_data[['Outcome']]
    x_test = test_data[feature_cols]
    y_test = test_data[['Outcome']]
    DT_2 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=9, class_weight={0: 1, 1: alpha})
    DT_2 = DT_2.fit(x_train, y_train)
    y_pred = DT_2.predict(x_test)
    print("Confusion matrix of DT_2:\n{}\n".format(metrics.confusion_matrix(y_test, y_pred)))
    # open csv file for tree result
    # f = open("experiments.csv", 'w')
    # for i in [3, 9, 27]:
    #     DT_2 = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=i)
    #     DT_2 = DT_2.fit(x_train, y_train)
    #     y_pred = DT_2.predict(x_test)
    #     f.write("{}, {}\n".format(i, metrics.accuracy_score(y_test, y_pred)))
    #     print("Confusion matrix for {}:\n{}\n".format(i, metrics.confusion_matrix(y_test, y_pred)))
    # tree_data = tree.export_graphviz(DT_2)
    # Draw graph
    # graph = pydotplus.graph_from_dot_data(tree_data)
    # Show graph
    # Image(graph.create_png())
    # graph.write_png("DT.png")
    # f.close()
