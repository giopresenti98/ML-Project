import pandas as pd
import Data
import Plot
from Classifiers import knn, svm, kde
import os
from sklearn.metrics import accuracy_score

n_train = 1000
# Load the dataset
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
print(data)
(xtr, ytr, xts, yts)=Data.feature_extractor(data,n_train)



#print(xtr)
#print(xtr.shape)

# Train the model
clf = knn.knn_train(xtr, ytr)
clf2 = svm.svm_train(xtr, ytr)
#clf3 = kde.kde_train(xtr, ytr)

# Plot the decision regions
Plot.plot_decision_regions(clf, xtr, ytr, resolution=0.1)
Plot.plot_decision_regions(clf2, xtr, ytr, resolution=0.1)
#Plot.plot_decision_regions(clf3, xtr, ytr, resolution=0.1)

print("Accuracy of KNN: ", accuracy_score(yts, clf.predict(xts)))
print("Accuracy of SVM: ", accuracy_score(yts, clf2.predict(xts)))
#print("Accuracy of KDE: ", accuracy_score(yts, clf3.predict(xts)))
# 