from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
import pandas as pd
import data as dt
import plot

import os

import time
from sklearn.metrics import accuracy_score


# Load the dataset

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'Dataset/data.csv')
data = pd.read_csv(filename, on_bad_lines='skip')
# print(data)

n_train = 500
# n_test = (len(data)-n_train)
n_test = 50000

xtr, ytr, xts, yts = dt.feature_extractor(data, n_train, n_test)


# Define the classifiers
KNN_clf = KNeighborsClassifier(n_neighbors=6)
SVM_clf = SVC(kernel="rbf")
DT_clf = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=0)
DT_reg = DecisionTreeRegressor()
GNB_clf = GaussianNB()
SGD_reg = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)

# List of classifiers to be trained and tested
clfs = [KNN_clf, SVM_clf, DT_clf, DT_reg, GNB_clf, SGD_reg]

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report    
# Function to plot the confusion matrix
def plot_confusion_matrix(y_test,y_pred,classifier_name):
    plt.figure(figsize=(5,5))
    plt.title('Confusion Matrix of ' + classifier_name)
    cm=confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True,fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

# Function to plot the ROC AUC curve
def plot_roc_auc_curve(y_test, y_pred, classifier_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred) # Calculate the FPR and TPR
    auc = roc_auc_score(y_test, y_pred) # Calculate the AUC
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ' + classifier_name)
    plt.legend(loc="lower right")
    plt.show()

# Histogram of the features and accuracy
for clf in clfs:

    # For every classifier, plot the feature importance histogram...
    print('\n')

    print(f"---\t{clf}\t---")
    tic = time.perf_counter()
    clf.fit(xtr, ytr)
    toc = time.perf_counter()
    print(f'Training time:\t {toc-tic:.4f} seconds')

    tic = time.perf_counter()
    predictions = clf.predict(xts)
    toc = time.perf_counter()
    print(f'Prediction time: {toc-tic:.4f} seconds')

    print(f'Accuracy score:\t {accuracy_score(yts, predictions):.4f}')
    
    
    results = permutation_importance(clf, xts, yts, scoring='accuracy')
    importances = results.importances_mean
    print('Feature importance:')

    print(f'Password length: {importances[0]:.4f}')
    print(f'Numbers in password: {importances[1]:.4f}')
    print(f'Symbols in password: {importances[2]:.4f}')
    print(f'Uppercase letters in password: {importances[3]:.4f}')

    plot.feature_importance_histogram(importances, title=f'{clf}')
    

"""
# Print the classification report and plot the accuracy, 
# confusion matrix and ROC AUC curve for each classifier
for index, y_pred in enumerate(predictions_list):
    print("Now showing results for",classifier_name_list[index])
    print(classification_report(y_test, y_pred))
    plot_accuracy(y_test, y_pred,classifier_name_list[index])
    plot_confusion_matrix(y_test, y_pred,classifier_name_list[index])
    plot_roc_auc_curve(y_test, y_pred, classifier_name_list[index])"""
# Plot the dataset with decision regions

# Plot.plot_decision_regions(clf, xts, yts, resolution=0.1, title='KNN')
# Plot.plot_decision_regions(clf2, xtr, ytr, resolution=0.1, title='SVM')
# Plot.plot_decision_regions(clf3, xtr, ytr, resolution=0.1, title='DT')
# Plot.plot_decision_regions(clf4, xtr, ytr, resolution=0.1, title='GNB')
