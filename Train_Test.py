import time
import Plot
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

def train_and_test(clf, xtr, xts, ytr, yts):
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

    results = permutation_importance(clf, xtr, ytr, scoring='accuracy')
    importances = results.importances_mean
    print(f'Feature importance: {importances}')

    Plot.feature_importance_histogram(importances, title=f'{clf}')