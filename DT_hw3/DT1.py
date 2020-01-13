from sklearn.datasets import load_breast_cancer
from sklearn import tree, metrics
import pandas as pd
import numpy as np

from id3 import Id3Estimator
from id3 import export_graphviz

import os
import csv
import shutil
import warnings
from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import hashlib

from sklearn.utils import check_random_state

# from ..utils import Bunch
# from ..utils import check_random_state



from urllib.request import urlretrieve

RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])

# def load_data():
#     for line in open(train.csv, 'r+'):


def load_data(module_path, data_file_name):
    with open(module_path + data_file_name) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(568)
        n_features = len(temp)
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        data[0] = np.asarray(temp[:-1], dtype=np.float64)
        target[0] = np.asarray(temp[-1], dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i+1] = np.asarray(ir[:-1], dtype=np.float64)
            target[i+1] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names

def load_tree_data(return_X_y=False):
    module_path = "./"  # dirname(__file__)
    data, target, target_names = load_data(module_path, 'train.csv')
    csv_filename = join(module_path, 'train.csv')

    with open('train.csv') as csv_file:
        ftrain = csv_file.read()

    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    feature_names = np.array(['Pregnancies', 'Glucose',
                              'BloodPressure', 'SkinThickness',
                              'Insulin', 'BMI',
                              'DiabetesPedigreeFunction', 'Age'])

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=ftrain,
                 feature_names=feature_names,
                 filename=csv_filename)



if __name__ == '__main__':
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    DT_ID3 = tree.DecisionTreeClassifier(criterion="entropy")
    train_data = pd.read_csv('train.csv', header=None, names=col_names)
    test_data = pd.read_csv('test.csv', header=None, names=col_names)
    train_data.head()
    test_data.head()
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    x_train = train_data[feature_cols]
    y_train = train_data[['Outcome']]
    x_test = test_data[feature_cols]
    y_test = test_data[['Outcome']]
    DT_ID3 = DT_ID3.fit(x_train, y_train)
    y_pred = DT_ID3.predict(x_test)
    print("Accuracy:\n", metrics.accuracy_score(y_test, y_pred), "\n")
    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred), "\n")
    print("BP")
