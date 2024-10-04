import numpy as np

def feature_extractor(data,nRows):
    # Shuffle the dataset
    data_shuffled = data.sample(frac=1)

    # n_samples=int(len(data['password']))

    # Generate features
    pass_length = (data_shuffled['password'].str.len().head(nRows).values).T
    numeric = (data_shuffled['password'].head(nRows).apply(lambda x: len([str(x) for x in list(x) if str(x).isdigit()]))).values.T
    # Split the dataset into features and target variable
    xtr = np.array([pass_length, numeric]).T
    ytr = data_shuffled['strength'].head(nRows).values
    return (xtr, ytr)