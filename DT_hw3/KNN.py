from operator import itemgetter
from sklearn import metrics
import pandas as pd
import numpy as np


def normalize_data(training, testing):
    min_vec = np.amin(training, axis=0)
    max_vec = np.amax(training, axis=0)
    diff = max_vec - min_vec
    normal_train = [(x - min_vec)/diff for x in training]
    normal_test = [(x - min_vec)/diff for x in testing]
    return np.asarray(normal_train), np.asarray(normal_test)


def KNN(DB, test_group, K):

    prediction = []
    for sample in test_group:
        K_closest = []
        for db_point in DB:
            cur_dist = np.linalg.norm(sample - db_point[:-1])
            most_remote = max(K_closest, key=itemgetter(0))[0] if len(K_closest) >= K else 0
            if len(K_closest) < K or cur_dist < most_remote:
                if len(K_closest) >= K:
                    rm_elemnt = [item for item in K_closest if item[0] == most_remote]
                    K_closest.remove(rm_elemnt[0])
                K_closest.append((cur_dist, db_point))
        res = sum([x[1] for x in K_closest])
        if res[-1] > int(K/2):
            prediction.append(1)
        else:
            prediction.append(0)
    return np.asarray(prediction)

    pass


if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_np = train_df.to_numpy()
    test_np = test_df.to_numpy()

    norm_train, norm_test = normalize_data(train_np, test_np)

    y_test = test_np[:, -1]
    y_pred = KNN(norm_train, norm_test[:, :-1], 9)
    mat = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", mat, "\n")
    # print("err_w: ", 4 * mat[1, 0] + mat[0, 1])

    # print("f1:\n", rep, "\n")
