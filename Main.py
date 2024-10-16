import pandas as pd
import Data
import Plot
from Classifiers import knn, svm, gnb, dt
import os
from sklearn.metrics import accuracy_score
import time

# Load the dataset

n_train = 40000
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
print(data)
xtr, ytr, xts, yts = Data.feature_extractor(data, n_train)

# Train the model

KNeigbours = knn.knn_train(xtr, ytr)
SupportVector = svm.svm_train(xtr, ytr)
DecisionTree = dt.dt_train(xtr, ytr)
Gaussian = gnb.gnb_train(xtr, ytr)

#Accuracy of the models...

#print('KNN Accuracy: ', accuracy_score(yts, clf.predict(xts)))
#print('SVM Accuracy: ', accuracy_score(yts, clf2.predict(xts)))
#print('DT Accuracy: ', accuracy_score(yts, clf3.predict(xts)))
#print('GNB Accuracy: ', accuracy_score(yts, clf4.predict(xts)))

#Histogram of the features...

def plot_histogram():

    clfs = [KNeigbours, SupportVector, DecisionTree, Gaussian]
    for clf in clfs:
        print(f'Accuracy of: {clf}', accuracy_score(yts, clf.predict(xts)))
        tick = time.perf_counter()
        Plot.feature_importance_histogram(clf, xtr, ytr, title=f'{clf}')
        tock = time.perf_counter()
        print('Elapsed Time: ', tock-tick)

plot_histogram()

#tick1 = time.perf_counter()
#Plot.feature_importance_histogram(clf2, xtr, ytr, title='SVM')
#tock2 = time.perf_counter()
#print('Elapsed Time: ', tock2-tick1)
#tick3 = time.perf_counter()
#Plot.feature_importance_histogram(clf3, xtr, ytr, title='DT')
#tock3 = time.perf_counter()
#print('Elapsed Time: ', tock3-tick3)
#tick4 = time.perf_counter()
#Plot.feature_importance_histogram(clf4, xtr, ytr, title='GNB')
#tock4 = time.perf_counter()
#print('Elapsed Time: ', tock4-tick4)

#Plot the dataset with decision regions...

#Plot.plot_decision_regions(clf, xts, yts, resolution=0.1, title='KNN, Accuracy: ' + str(accuracy_score(yts, clf.predict(xts))))
#Plot.plot_decision_regions(clf2, xtr, ytr, resolution=0.1, title='SVM, Accuracy: ' + str(accuracy_score(yts, clf2.predict(xts))))
#Plot.plot_decision_regions(clf3, xtr, ytr, resolution=0.1, title='DT, Accuracy: ' + str(accuracy_score(yts, clf3.predict(xts))))
#Plot.plot_decision_regions(clf4, xtr, ytr, resolution=0.1, title='GNB, Accuracy: ' + str(accuracy_score(yts, clf4.predict(xts))))
 