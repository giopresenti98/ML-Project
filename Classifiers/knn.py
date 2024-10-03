from sklearn.neighbors import KNeighborsClassifier

def train (xtr,ytr):
    clf=KNeighborsClassifier(n_neighbors=1)
    clf.fit(xtr,ytr)
    return clf
