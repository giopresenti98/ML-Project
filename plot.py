from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_dataset(x, y):
    colors = ['kx', 'bo', 'r.', 'g+', 'y', 'm', 'c']
    classes = np.unique(y)
    for i, k in enumerate(classes):
        plt.plot(x[y == k, 0], x[y == k, 1], colors[i])

# Plot the dataset...


def plot_decision_regions(classifier, x, y, resolution=0.02, title=None, accuracy=None):

    # setup marker generator and color map
    colors = ('black', 'blue', 'red', 'lightgreen',
              'yellow', 'magenta', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    x2_min, x2_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plot_dataset(x, y)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.xlabel('Lenght of Password')
    plt.ylabel('Numbers in Password')
    plt.title(title, accuracy)
    plt.show()
    return


def feature_importance_histogram(importances, title=None):
    colors = ['black', 'blue', 'red',
              'lightgreen', 'yellow', 'magenta', 'cyan']

    plt.bar([x for x in range(len(importances))], importances,
            tick_label=[x for x in range(len(importances))],
            color=colors, edgecolor='black', linewidth=1.2, alpha=0.7)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.title(title)
    plt.show()
  
