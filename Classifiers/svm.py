from sklearn.svm import SVC

def svm_train(xtr, ytr):
    svm = SVC(kernel="rbf")
    svm.fit(xtr, ytr)
    return svm

