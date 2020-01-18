from sklearn import tree, metrics
import pandas as pd
import numpy as np

if __name__ == '__main__':
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    train_np = train_df.to_numpy()
    test_np = test_df.to_numpy()

    neg_bool_arr = (train_np[:, -1] == 0)
    pos_bool_arr = (train_np[:, -1] == 1)
    pos_num = np.count_nonzero(pos_bool_arr)
    neg_idx = np.where(neg_bool_arr)[0]
    neg_idx_rm = neg_idx[pos_num:]
    train_np = np.delete(train_np, neg_idx_rm, 0)

    neg_num = np.count_nonzero(neg_bool_arr)

    x_train, y_train = train_np[:, :-1], train_np[:, -1]
    x_test, y_test = test_np[:, :-1], test_np[:, -1]

    tree_classifier = tree.DecisionTreeClassifier(criterion='entropy')
    tree_classifier.fit(x_train, y_train)
    y_pred = tree_classifier.predict(x_test)
    print("Confusion matrix:\n{}\n".format(metrics.confusion_matrix(y_test, y_pred)))

