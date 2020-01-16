from KNN import normalize_data, KNN
import pandas as pd
import numpy as np



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
    flag = 0
    while True:
        for i in range(8):
            if i not in best_sub:
                subset = best_sub + (i,)
            else:
                continue
            # subset = (1,7,6,5,2)
            train_tmp = norm_train[:, (subset + (8,))]
            test_tmp = norm_test[:, subset]
            y_pred = KNN(train_tmp, test_tmp, 9)
            # acc = acc_calc(y_test, y_pred)
            acc = float(np.sum(y_test == y_pred) / len(y_test))
            # print("for element " + str(subset) +" recieved acc of " + str(acc) + "\n")
            if acc >= max_acc:
                flag = 1
                max_acc = acc
                tmp_sub = subset
        best_sub = tmp_sub
        if flag == 0:
            break
        flag = 0
    print(list(best_sub))
    print(max_acc)


