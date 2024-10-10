import pandas as pd
import Data
import Plot
from Classifiers import knn, svm, kde
import os
from sklearn.metrics import accuracy_score

trainSamples = 10000

# Load the dataset
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
print(data)
(xtr,ytr)=Data.feature_extractor(data,trainSamples)



#print(xtr)
#print(xtr.shape)

# Train the model
clf = knn.knn_train(xtr, ytr)
clf2 = svm.svm_train(xtr, ytr)
clf3 = kde.kde_train(xtr, ytr)

# Plot the decision regions
Plot.plot_decision_regions(clf, xtr, ytr, resolution=0.1)
Plot.plot_decision_regions(clf2, xtr, ytr, resolution=0.1)
Plot.plot_decision_regions(clf3, xtr, ytr, resolution=0.1)
