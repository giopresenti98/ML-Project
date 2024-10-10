import numpy as np
from sklearn.neighbors import KernelDensity


class CClassifierKDE:
    """
    Class implementing a KDE classifier
    """

    def __init__(self, bandwidth=1.0, kernel='gaussian', metric='euclidean', posterior=True):
        self._priors = None
        self._bandwidth = bandwidth
        self._kernel = kernel
        self._metric = metric
        self._kde = None
        # if True, return posterior probabilities; otherwise returns joint probs
        self._posterior = posterior
        return


    @property
    def posterior(self):
        return self._posterior

    @posterior.setter
    def posterior(self, value):
        self._posterior = bool(value)


    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, priors):
        """Set priors (if not estimated from training data)."""
        self._priors = np.array(priors)

    def fit(self, x, y):
        """Estimate priors and fit one KDE per class."""
        n_classes = np.unique(y).size
        n_features = x.shape[1]
        self._kde = [KernelDensity(
            bandwidth=self._bandwidth, kernel=self._kernel,
            metric=self._metric) for k in range(n_classes)]

        self._priors = np.zeros(shape=(n_classes,))

        for k in range(n_classes):

            self._kde[k].fit(x[y==k,:])
            self._priors[k] = (y == k).mean()

        self._priors /= self._priors.sum()  # ensure priors sum up to 1
        return self

    def decision_function(self, x):
        """Return posterior or joint probability estimates for each class,
        depending on whether posterior=True or False.
        """
        n_samples = x.shape[0]
        n_classes = len(self._kde)
        scores = np.zeros(shape=(n_samples, n_classes))
        for k in range(n_classes):
            likelihood_k = np.exp(self._kde[k].score_samples(x))  # we use exp as score_samples returns the log likelihood
            scores[:, k] = self._priors[k] * likelihood_k  # joint probability

        if self.posterior:
            # if posterior probs are required, divide joint probs by evidence
            evidence = scores.sum(axis=1)
            for k in range(n_classes):
                # normalize per row to estimate posterior
                scores[:, k] /= evidence
        return scores

    def predict(self, x):
        """Return predicted labels."""
        scores = self.decision_function(x)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
    

def kde_train (xtr,ytr):
    kde = CClassifierKDE()
    kde.fit(xtr,ytr)
    return kde