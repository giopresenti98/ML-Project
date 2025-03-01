from Classifiers import knn, svm, gnb, dt
from sklearn.metrics import accuracy_score
import pandas as pd
import Data
import Plot
import os
import time

# Load the dataset

n_train = 40000
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
print(data)
xtr, ytr, xts, yts = Data.feature_extractor(data, n_train)

# Train the model
tic = time.perf_counter()
KNeigbours = knn.knn_train(xtr, ytr)
toc = time.perf_counter()
print(f'----Elapsed Time for training KNN: ', toc-tic)

tic = time.perf_counter()
SupportVector = svm.svm_train(xtr, ytr)
toc = time.perf_counter()
print(f'----Elapsed Time for training SVM: ', toc-tic)

tic = time.perf_counter()
DecisionTreeC = dt.dtc_train(xtr, ytr)
toc = time.perf_counter()
print(f'----Elapsed Time for training DTC: ', toc-tic)

tic = time.perf_counter()
Gaussian = gnb.gnb_train(xtr, ytr)
toc = time.perf_counter()
print(f'----Elapsed Time for training GNB: ', toc-tic)

tic = time.perf_counter()
DecisionTreeR = dt.dtr_train(xtr, ytr)
toc = time.perf_counter()
print(f'----Elapsed Time for training DTR: ', toc-tic)

clfs = [KNeigbours, SupportVector, DecisionTreeC, DecisionTreeR, Gaussian]

# Histogram of the features and accuracy...

Plot.results(clfs, xtr, xts, ytr, yts)

# Plot the dataset with decision regions...

# Plot.plot_decision_regions(clf, xts, yts, resolution=0.1, title='KNN, Accuracy: ' + str(accuracy_score(yts, clf.predict(xts))))
# Plot.plot_decision_regions(clf2, xtr, ytr, resolution=0.1, title='SVM, Accuracy: ' + str(accuracy_score(yts, clf2.predict(xts))))
# Plot.plot_decision_regions(clf3, xtr, ytr, resolution=0.1, title='DT, Accuracy: ' + str(accuracy_score(yts, clf3.predict(xts))))
# Plot.plot_decision_regions(clf4, xtr, ytr, resolution=0.1, title='GNB, Accuracy: ' + str(accuracy_score(yts, clf4.predict(xts))))
