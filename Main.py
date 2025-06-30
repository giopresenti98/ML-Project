from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import pandas as pd
import data as dt
import plot
from train_and_test import train_and_test
import os


# Load the dataset

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
# print(data)

n_train = 20000
# n_test = (len(data)-n_train)
n_test = 20000

xtr, ytr, xts, yts = dt.feature_extractor(data, n_train, n_test)


# Define the classifiers
KNN_clf = KNeighborsClassifier(n_neighbors=6)
SVM_clf = SVC(kernel="rbf")
DT_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
DT_reg = DecisionTreeRegressor()
GNB_clf = GaussianNB()
SGD_reg = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

# List of classifiers to be trained and tested
clfs = [KNN_clf, SVM_clf, DT_clf, DT_reg, GNB_clf, SGD_reg]

# Histogram of the features and accuracy
for clf in clfs:
    
    train_and_test(clf, xtr, xts, ytr, yts)
    
    results = permutation_importance(clf, xts, yts, scoring='accuracy')
    importances = results.importances_mean
    print(f'Feature importance: {importances}')

    plot.feature_importance_histogram(importances, title=f'{clf}')


# Plot the dataset with decision regions

# Plot.plot_decision_regions(clf, xts, yts, resolution=0.1, title='KNN')
# Plot.plot_decision_regions(clf2, xtr, ytr, resolution=0.1, title='SVM')
# Plot.plot_decision_regions(clf3, xtr, ytr, resolution=0.1, title='DT')
# Plot.plot_decision_regions(clf4, xtr, ytr, resolution=0.1, title='GNB')
