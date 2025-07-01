from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_dataset(x, y, bins=20, title="Password Length Distribution by Strength"):
    """
    Plots the distribution of password lengths grouped by strength as grouped bars.
    x: feature matrix (password length should be the first column)
    y: strength labels
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
    plt.figure(figsize=(5, 5))
    plt.title("Confusion Matrix of " + classifier_name)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.show()


def feature_importance_histogram(importances, title=None):
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
