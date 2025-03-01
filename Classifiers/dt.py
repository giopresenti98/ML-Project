from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

def dtc_train(xtr, ytr):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    clf.fit(xtr, ytr)
    return clf

def dtr_train(xtr, ytr):
    clf = DecisionTreeRegressor()
    clf.fit(xtr, ytr)
    return clf