import numpy as np


def feature_extractor(data, n_train):
    # Shuffle the dataset
    data_shuffled = data#.sample(frac=1)
    n_test = (len(data)-n_train)
    
    data_shuffled['password'] = data_shuffled['password'].astype(str)
    
    # Generate features
    pass_length_train = (data_shuffled['password'].str.len().head(
        n_train).values).T
    numeric_train = (data_shuffled['password'].head(n_train).apply(
        lambda x: len([str(x) for x in list(x) if str(x).isdigit()]))).values.T
    letter_train = (data_shuffled['password'].head(n_train).apply(
        lambda x: len([str(x) for x in list(x) if str(x).isalpha()]))).values.T
    upper_train = (data_shuffled['password'].head(n_train).apply(
        lambda x: len([str(x) for x in list(x) if str(x).isupper()]))).values.T


    pass_length_test = (data_shuffled['password'].str.len().tail(
        n_test).values).T
    numeric_test = (data_shuffled['password'].tail(n_test).apply(
        lambda x: len([str(x) for x in list(x) if str(x).isdigit()]))).values.T
    letter_test = (data_shuffled['password'].tail(n_test).apply(
        lambda x: len([str(x) for x in list(x) if str(x).isalpha()]))).values.T
    upper_test = (data_shuffled['password'].tail(n_test).apply(
        lambda x: len([str(x) for x in list(x) if str(x).isupper()]))).values.T
    

    # Split the dataset into features and target variable
    xtr = np.array([pass_length_train, numeric_train, letter_train, upper_train]).T
    ytr = data_shuffled['strength'].head(n_train).values
    
    xts = np.array([pass_length_test, numeric_test, letter_test, upper_test]).T
    yts = data_shuffled['strength'].tail(n_test).values

    return (xtr, ytr, xts, yts)
