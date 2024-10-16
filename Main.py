import pandas as pd
import Data
import Plot
from Classifiers import knn, svm, gnb, dt
import os
from sklearn.metrics import accuracy_score

# Load the dataset

n_train = 400000
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
print(data)
xtr, ytr, xts, yts = Data.feature_extractor(data,n_train)


# Train the model

clf = knn.knn_train(xtr, ytr)
clf2 = svm.svm_train(xtr, ytr)
clf3 = dt.dt_train(xtr, ytr)
clf4 = gnb.gnb_train(xtr, ytr)

#Accuracy of the models...

print('KNN Accuracy: ', accuracy_score(yts, clf.predict(xts)))
print('SVM Accuracy: ', accuracy_score(yts, clf2.predict(xts)))
print('DT Accuracy: ', accuracy_score(yts, clf3.predict(xts)))
print('GNB Accuracy: ', accuracy_score(yts, clf4.predict(xts)))

#Plot the dataset in three dimensions, test...

#Plot.plot_dataset_threeD(clf, xtr, ytr, title='KNN, Accuracy: ' + str(accuracy_score(yts, clf.predict(xts))))
#Plot.plot_dataset_threeD(clf2, xtr, ytr, title='SVM, Accuracy: ' + str(accuracy_score(yts, clf2.predict(xts))))
#Plot.plot_dataset_threeD(clf3, xtr, ytr, title='DT, Accuracy: ' + str(accuracy_score(yts, clf3.predict(xts))))
#Plot.plot_dataset_threeD(clf4, xtr, ytr, title='GNB, Accuracy: ' + str(accuracy_score(yts, clf4.predict(xts))))

#Histogram of the features...

#Plot.feature_importance_histogram(clf, xtr, ytr)
#Plot.feature_importance_histogram(clf2, xtr, ytr)
#Plot.feature_importance_histogram(clf3, xtr, ytr)
#Plot.feature_importance_histogram(clf4, xtr, ytr)

#Plot the dataset with decision regions...

#Plot.plot_decision_regions(clf, xts, yts, resolution=0.1, title='KNN, Accuracy: ' + str(accuracy_score(yts, clf.predict(xts))))
#Plot.plot_decision_regions(clf2, xtr, ytr, resolution=0.1, title='SVM, Accuracy: ' + str(accuracy_score(yts, clf2.predict(xts))))
#Plot.plot_decision_regions(clf3, xtr, ytr, resolution=0.1, title='DT, Accuracy: ' + str(accuracy_score(yts, clf3.predict(xts))))
#Plot.plot_decision_regions(clf4, xtr, ytr, resolution=0.1, title='GNB, Accuracy: ' + str(accuracy_score(yts, clf4.predict(xts))))
 