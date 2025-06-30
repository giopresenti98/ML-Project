from sklearn import naive_bayes
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import pandas as pd
import Data
import Plot
import os


# Load the dataset


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
# print(data)

n_train = 20000
#n_test = (len(data)-n_train)
n_test = 20000

xtr, ytr, xts, yts = Data.feature_extractor(data, n_train, n_test)


# Train the model
KNN_clf = KNeighborsClassifier(n_neighbors=6)
SVM_clf = SVC(kernel="rbf")
DT_clf = clf = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=0)
DT_reg = DecisionTreeRegressor()
GNB_clf = naive_bayes.GaussianNB()
SGD_reg = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

clfs = [KNN_clf, SVM_clf, DT_clf, DT_reg, GNB_clf,SGD_reg]

# Histogram of the features and accuracy...

Plot.train_and_test(clfs, xtr, xts, ytr, yts)

# Plot the dataset with decision regions...

# Plot.plot_decision_regions(clf, xts, yts, resolution=0.1, title='KNN')
# Plot.plot_decision_regions(clf2, xtr, ytr, resolution=0.1, title='SVM')
# Plot.plot_decision_regions(clf3, xtr, ytr, resolution=0.1, title='DT')
# Plot.plot_decision_regions(clf4, xtr, ytr, resolution=0.1, title='GNB')
