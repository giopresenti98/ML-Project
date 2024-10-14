from sklearn.tree import DecisionTreeClassifier

def dt_train(xtr, ytr):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    clf.fit(xtr, ytr)
    return clf