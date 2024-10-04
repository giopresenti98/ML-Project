import pandas as pd


import Data
import Plot
from Classifiers import knn
import os

trainSamples = 10000

# Load the dataset
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
print(data)
(xtr,ytr)=Data.feature_extractor(data,trainSamples)

print(xtr)
print(xtr.shape)

# Train the model
clf = knn.train(xtr, ytr)

# Plot the decision regions
Plot.plot_decision_regions(clf, xtr, ytr, resolution=0.1)
