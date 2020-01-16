from sklearn import tree, metrics
import pandas as pd
import numpy as np
import random


def change_pred(y_pred):
    prob = [0.05, 0.1, 0.2]
    y = []
    for p in prob:
        y.append([1 if (i == 0 and random.random() < p) else i for i in y_pred])

    return y


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_np = train_df.to_numpy()
    test_np = test_df.to_numpy()

    x_train, y_train = train_np[:, :-1], train_np[:, -1]
    x_test, y_test = test_np[:, :-1], test_np[:, -1]

    tree_classifier = tree.DecisionTreeClassifier(criterion='entropy')
    tree_classifier.fit(x_train, y_train)
    y_pred = tree_classifier.predict(x_test)
    mat = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", mat, "\n")
    print("err_w: ", 4 * mat[1, 0] + mat[0, 1])
    y_fix = change_pred(y_pred)
    for y_ans in y_fix:
        mat = metrics.confusion_matrix(y_test, y_ans)
        print("Confusion matrix:\n", mat, "\n")
        print("err_w: ", 4*mat[1, 0] + mat[0, 1])

