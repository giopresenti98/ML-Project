from sklearn.neighbors import KNeighborsClassifier


clf=KNeighborsClassifier(n_neighbors=1)
clf.fit(xtr,ytr)
