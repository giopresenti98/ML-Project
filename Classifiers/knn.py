from sklearn.neighbors import KNeighborsClassifier

def train (xtr,ytr):
    clf=KNeighborsClassifier(n_neighbors=6)
    clf.fit(xtr,ytr)
    return clf
