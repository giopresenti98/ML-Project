import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
# Load the dataset
data = pd.read_csv('Python\\4-Machine learning\Project\ML-Project\Dataset\data.csv', on_bad_lines='skip')

pass_length = []
for i in data['password'].values:
    pass_length.append(len(str(i)))



# Split the dataset into features and target variable
xtr = np.array(pass_length).reshape(1, -1).T
ytr = data['strength'].values



clf=KNeighborsClassifier(n_neighbors=1)
clf.fit(xtr,ytr)


from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt

def plot_dataset(x, y): 
    colors = ['kx', 'bo', 'r.', 'g+', 'y', 'm', 'c']
    classes = np.unique(y)
    for i, k in enumerate(classes):
        plt.plot(x[y==k,0], x[y==k,1], colors[i])


def plot_decision_regions(classifier, x, y, resolution=0.02):
    # setup marker generator and color map
    colors = ('black', 'blue', 'red', 'lightgreen', 'yellow', 'magenta', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    x2_min, x2_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plot_dataset(x,y)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    return


plot_decision_regions(clf, xtr, ytr, resolution=0.1)