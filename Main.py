import pandas as pd
import Data
import Plot
from Classifiers import knn, svm, gnb, dt
import os
from sklearn.metrics import accuracy_score


# Load the dataset
n_train = 500000
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
(xtr, ytr, xts, yts)=Data.feature_extractor(data,n_train)

# Train the model

clf = knn.knn_train(xtr, ytr)
clf2 = svm.svm_train(xtr, ytr)
clf3 = dt.dt_train(xtr, ytr)
clf4 = gnb.gnb_train(xtr, ytr)
print("Accuracy of KNN: ", accuracy_score(yts, clf.predict(xts)))
print("Accuracy of SVM: ", accuracy_score(yts, clf2.predict(xts)))
print("Accuracy of DT: ", accuracy_score(yts, clf3.predict(xts)))
print("Accuracy of GNB: ", accuracy_score(yts, clf4.predict(xts)))

# Plot the decision regions

#Plot.plot_decision_regions(clf, xts, yts, resolution=0.1, title='KNN, Accuracy: ' + str(accuracy_score(yts, clf.predict(xts))))
#Plot.plot_decision_regions(clf2, xtr, ytr, resolution=0.1)
#Plot.plot_decision_regions(clf3, xtr, ytr, resolution=0.1)

#print("Accuracy of KNN: ", accuracy_score(yts, clf.predict(xts)))
#print("Accuracy of SVM: ", accuracy_score(yts, clf2.predict(xts)))
#print("Accuracy of KDE: ", accuracy_score(yts, clf3.predict(xts)))
 