from sklearn.datasets import load_breast_cancer
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

import numpy as np

from urllib.request import urlretrieve

RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])

# def load_data():
#     for line in open(train.csv, 'r+'):


def load_data(module_path, data_file_name):
    """Loads data from module_path/data/data_file_name.

    Parameters
    ----------
    module_path : string
        The module path.

    data_file_name : string
        Name of csv file to be loaded from
        module_path/data/data_file_name. For example 'wine_data.csv'.

    Returns
    -------
    data : Numpy array
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    target : Numpy array
        A 1D array holding target variables for all the samples in `data.
        For example target[0] is the target varible for data[0].

    target_names : Numpy array
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.
    """
    with open(module_path + data_file_name) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names

def load_tree_data(return_X_y=False):
    """Load and return the breast cancer wisconsin dataset (classification).

    The breast cancer dataset is a classic and very easy binary classification
    dataset.

    =================   ==============
    Classes                          2
    Samples per class    212(M),357(B)
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <breast_cancer_dataset>`.

    Parameters
    ----------
    return_X_y : boolean, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the full description of
        the dataset, 'filename', the physical location of
        breast cancer csv dataset (added in version `0.20`).

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is
    downloaded from:
    https://goo.gl/U2Uwz2

    Examples
    --------
    Let's say you are interested in the samples 10, 50, and 85, and want to
    know their class name.

    >>> from sklearn.datasets import load_breast_cancer
    >>> data = load_breast_cancer()
    >>> data.target[[10, 50, 85]]
    array([0, 1, 0])
    >>> list(data.target_names)
    ['malignant', 'benign']
    """
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
    # bunch = load_breast_cancer()
    bunch = load_tree_data()
    estimator = Id3Estimator()
    estimator.fit(bunch.data, bunch.target)
    export_graphviz(estimator.tree_, 'tree.dot', bunch.feature_names)