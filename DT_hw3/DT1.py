from sklearn import tree, metrics
import pandas as pd
import numpy as np

# if __name__ == '__main__':
#     train_df = pd.read_csv('train.csv')
#     test_df = pd.read_csv('test.csv')
#     train_np = train_df.to_numpy()
#     test_np = test_df.to_numpy()
#
#     x_train, y_train = train_np[:, :-1], train_np[:, -1]
#     x_test, y_test = test_np[:, :-1], test_np[:, -1]
#
#     tree_classifier = tree.DecisionTreeClassifier(criterion='entropy')
#     tree_classifier.fit(x_train, y_train)
#     y_pred = tree_classifier.predict(x_test)
#     print("Accuracy:\n", metrics.accuracy_score(y_test, y_pred), "\n")
#     print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred), "\n")
#     print("BP")

if __name__ == '__main__':
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    DT_ID3 = tree.DecisionTreeClassifier(criterion="entropy")
    train_data = pd.read_csv('train.csv', header=None, names=col_names)
    test_data = pd.read_csv('test.csv', header=None, names=col_names)
    # train_data.head()
    # test_data.head()
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    x_train = train_data[feature_cols]
    y_train = train_data[['Outcome']]
    x_test = test_data[feature_cols]
    y_test = test_data[['Outcome']]
    DT_ID3 = DT_ID3.fit(x_train, y_train)
    y_pred = DT_ID3.predict(x_test)
    # print("Accuracy:\n", metrics.accuracy_score(y_test, y_pred), "\n")
    print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred), "\n")
