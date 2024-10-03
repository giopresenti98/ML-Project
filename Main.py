import pandas as pd
import numpy as np


import Plot
from Classifiers import knn

nRows = 25000
# Load the dataset
data = pd.read_csv('Python\\4-Machine learning\Project\ML-Project\Dataset\data.csv',
                   on_bad_lines='skip', nrows=nRows*10)
data_shuffled=data.sample(frac=1)
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

clf = knn.train(xtr, ytr)

Plot.plot_decision_regions(clf, xtr, ytr, resolution=0.1)
