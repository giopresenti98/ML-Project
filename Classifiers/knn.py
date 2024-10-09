from sklearn.neighbors import KNeighborsClassifier

def knn_train (xtr,ytr):
    clf=KNeighborsClassifier(n_neighbors=6)
    clf.fit(xtr,ytr)
    return clf
