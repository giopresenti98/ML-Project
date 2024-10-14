from sklearn import naive_bayes
def gnb_train(xtr, ytr):
    gnb = naive_bayes.GaussianNB()
    gnb.fit(xtr, ytr)
    return gnb
