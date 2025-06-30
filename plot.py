from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_dataset(x, y):
    colors = ['kx', 'bo', 'r.', 'g+', 'y', 'm', 'c']
    classes = np.unique(y)
    for i, k in enumerate(classes):
        plt.plot(x[y == k, 0], x[y == k, 1], colors[i])
    plt.show()

# Function to plot the confusion matrix
def plot_confusion_matrix(y_test,y_pred,classifier_name):
    plt.figure(figsize=(5,5))
    plt.title('Confusion Matrix of ' + classifier_name)
    cm=confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True,fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


def feature_importance_histogram(importances, title=None):
    colors = ['black', 'blue', 'red',
              'lightgreen', 'yellow', 'magenta', 'cyan']

    plt.bar([x for x in range(len(importances))], importances,
            tick_label=[x for x in range(len(importances))],
            color=colors, edgecolor='black', linewidth=1.2, alpha=0.7)
    plt.xlabel('Feature')
    plt.ylabel('Importance score')
    plt.xticks([0, 1, 2, 3], ["Lenght", "Digits", "Symbols", "Uppercase letters"])
    plt.title(title)
    plt.show()

