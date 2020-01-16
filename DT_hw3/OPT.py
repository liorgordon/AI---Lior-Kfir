from KNN import normalize_data, KNN
import pandas as pd
import numpy as np
import itertools



if __name__ == '__main__':
    train_df = pd.read_csv('train.csv', header=None)
    test_df = pd.read_csv('test.csv', header=None)
    train_np = train_df.to_numpy()
    test_np = test_df.to_numpy()
    # print(len(test_np[0]))
    norm_train, norm_test = normalize_data(train_np, test_np)

    y_test = test_np[:, -1]
    property_list = [x for x in range(len(test_np[0, :-1]))]
    max_acc = 0
    best_sub = ()
    for L in range(1, len(property_list)+1):
        for subset in itertools.combinations(property_list, L):
            train_tmp = norm_train[:, (subset + (8,))]
            test_tmp = norm_test[:, subset]
            y_pred = KNN(train_tmp, test_tmp, 9)
            acc = np.sum(y_test == y_pred) / len(y_test)
            if acc > max_acc:
                max_acc = acc
                best_sub = subset
            # print(train_tmp[0])
    print(list(best_sub))