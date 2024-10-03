import pandas as pd
import numpy as np


import Plot
from Classifiers import knn
import os

nRows = 25000

# Load the dataset
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip', nrows=nRows*10)

# Shuffle the dataset
data_shuffled = data.sample(frac=1)

# n_samples=int(len(data['password']))

# Generate features
pass_length = (data_shuffled['password'].str.len().head(nRows).values).T
numeric = (data_shuffled['password'].apply(lambda x: len([str(x)
           for x in list(x) if str(x).isdigit()])).head(nRows).values).T
# Split the dataset into features and target variable
xtr = np.array([pass_length, numeric]).T
ytr = data_shuffled['strength'].head(nRows).values

print(xtr)
print(xtr.shape)

# Train the model
clf = knn.train(xtr, ytr)

# Plot the decision regions
Plot.plot_decision_regions(clf, xtr, ytr, resolution=0.1)
