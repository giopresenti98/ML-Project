from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import pandas as pd
import data as dt
import plot
import os
import time
from sklearn.metrics import accuracy_score

# Load the dataset
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
# print(data)

n_train = 20000
n_test = 50000

xtr, ytr, xts, yts = dt.feature_extractor(data, n_train, n_test)

# Define the classifiers
KNN_clf = KNeighborsClassifier(n_neighbors=6)
SVM_clf = SVC(kernel="rbf")
DT_clf = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=0)
DT_reg = DecisionTreeRegressor()
GNB_clf = GaussianNB()
SGD_reg = SGDClassifier(loss="hinge", penalty="l2")

# List of classifiers to be trained and tested
clfs = [KNN_clf, SVM_clf, DT_clf, DT_reg, GNB_clf, SGD_reg]

# Histogram of the features and accuracy
for clf in clfs:

    # For every classifier, plot the feature importance histogram...
    print('\n')
    print(f"---\t{clf}\t---")
    tic = time.perf_counter()
    clf.fit(xtr, ytr)
    toc = time.perf_counter()
    print(f'Training time:\t {toc-tic:.4f} seconds')

    tic = time.perf_counter()
    predictions = clf.predict(xts)
    toc = time.perf_counter()
    print(f'Prediction time: {toc-tic:.4f} seconds')

    print(f'Accuracy score:\t {accuracy_score(yts, predictions):.4f}')

    results = permutation_importance(clf, xts, yts, scoring='accuracy')
    importances = results.importances_mean
    print('Feature importance:')

    print(f'Password length: {importances[0]:.4f}')
    print(f'Digits in password: {importances[1]:.4f}')
    print(f'Symbols in password: {importances[2]:.4f}')
    print(f'Uppercase letters in password: {importances[3]:.4f}')

    plot.feature_importance_histogram(importances, title=f'{clf}')

# Plot the dataset with decision regions

plot.plot_decision_regions(KNN_clf, xts, yts, resolution=0.1, title='KNN')
plot.plot_decision_regions(SVM_clf, xtr, ytr, resolution=0.1, title='SVM')
plot.plot_decision_regions(DT_clf, xtr, ytr, resolution=0.1, title='DT')
plot.plot_decision_regions(GNB_clf, xtr, ytr, resolution=0.1, title='GNB')
