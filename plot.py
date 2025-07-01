from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_dataset(x, y, bins=20, title="Password Length Distribution by Strength"):
    """
    Plots the distribution of password lengths grouped by their strength categories.
    Parameters:
        x (np.ndarray): 2D array where the first column contains password lengths.
        y (np.ndarray): 1D array of class labels corresponding to password strength for each sample in x.
        bins (int, optional): Number of bins to use for the histogram. Default is 20.
        title (str, optional): Title of the plot. Default is "Password Length Distribution by Strength".
    Displays:
        A bar plot showing the distribution of password lengths for each strength class, with different colors and a legend.
    """
    
    plt.figure(figsize=(10, 6))
    classes = np.unique(y)
    colors = ["red", "orange", "green", "blue", "purple", "cyan", "magenta"]
    all_lengths = x[:, 0]
    bin_edges = np.histogram_bin_edges(all_lengths, bins=bins)
    width = (bin_edges[1] - bin_edges[0]) / (len(classes) + 1)
    legend_names = ["Weak", "Medium", "Strong"]
    for i, k in enumerate(classes):
        lengths = x[y == k, 0]
        hist, _ = np.histogram(lengths, bins=bin_edges)
        plt.bar(
            bin_edges[:-1] + i * width,
            hist,
            width=width,
            color=colors[i % len(colors)],
            align="edge",
            label=legend_names[i],
        )

    plt.xlabel("Password Length")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(title="Password Strength")
    plt.grid(axis="y", alpha=0.4)
    plt.minorticks_on()                               # Enable minor ticks
    plt.grid(which='minor', axis='y', alpha=0.2, linewidth=0.3)  # Minor grid
    plt.show()


# Function to plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred, classifier_name):
    """
    Plots the confusion matrix for a given set of true and predicted labels using a heatmap.

    Args:
        y_test (array-like): True labels of the test dataset.
        y_pred (array-like): Predicted labels from the classifier.
        classifier_name (str): Name of the classifier to display in the plot title.

    Returns:
        None: Displays the confusion matrix plot.
    """
    plt.figure(figsize=(5, 5))
    plt.title("Confusion Matrix of " + classifier_name)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.show()


def feature_importance_histogram(importances, title=None):
    """
    Plots a histogram of feature importances.
    Parameters
    ----------
    importances : list or array-like
        The importance scores for each feature. The length should match the number of features.
    title : str, optional
        The title of the plot. Default is None.
    Notes
    -----
    - The x-axis labels are fixed to ["Lenght", "Digits", "Symbols", "Uppercase letters"] for the first four features.
    - The bar colors are cycled from a predefined list.
    - Displays the plot using matplotlib's `plt.show()`.
    """
    colors = ["black", "blue", "red", "lightgreen", "yellow", "magenta", "cyan"]

    plt.bar(
        [x for x in range(len(importances))],
        importances,
        tick_label=[x for x in range(len(importances))],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.7,
    )
    plt.xlabel("Feature")
    plt.ylabel("Importance score")
    plt.xticks([0, 1, 2, 3], ["Lenght", "Digits", "Symbols", "Uppercase letters"])
    plt.title(title)
    plt.show()
