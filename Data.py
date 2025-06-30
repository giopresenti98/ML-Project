import numpy as np


def feature_extractor(data, n_train, n_test):
    # Shuffle the dataset
    data_shuffled = data.sample(frac=1)  

    data_shuffled['password'] = data_shuffled['password'].astype(str)

    # Generate features
    pass_length_train = data_shuffled['password'].str.len().head(n_train).values
    numeric_train = data_shuffled['password'].head(n_train).apply(
        lambda x: len([c for c in x if c.isdigit()])).values
    symbols_train = data_shuffled['password'].head(n_train).apply(
        lambda x: len([c for c in x if not c.isalnum()])).values
    upper_train = data_shuffled['password'].head(n_train).apply(
        lambda x: len([c for c in x if c.isupper()])).values
    
    # Normalize features by password length (avoid division by zero)
    numeric_train_norm = numeric_train / np.where(pass_length_train == 0, 1, pass_length_train)
    symbols_train_norm = symbols_train / np.where(pass_length_train == 0, 1, pass_length_train)
    upper_train_norm = upper_train / np.where(pass_length_train == 0, 1, pass_length_train)
    
    pass_length_test = data_shuffled['password'].str.len().tail(n_test).values
    numeric_test = data_shuffled['password'].tail(n_test).apply(
        lambda x: len([c for c in x if c.isdigit()])).values
    symbols_test = data_shuffled['password'].tail(n_test).apply(
        lambda x: len([c for c in x if not c.isalnum()])).values
    upper_test = data_shuffled['password'].tail(n_test).apply(
        lambda x: len([c for c in x if c.isupper()])).values
    
    numeric_test_norm = numeric_test / np.where(pass_length_test == 0, 1, pass_length_test)
    symbols_test_norm = symbols_test / np.where(pass_length_test == 0, 1, pass_length_test)
    upper_test_norm = upper_test / np.where(pass_length_test == 0, 1, pass_length_test)
    
    # Split the dataset into features and target variable
    #xtr = np.array([pass_length_train, numeric_train, symbols_train, upper_train]).T
    xtr = np.array([pass_length_train, numeric_train_norm, symbols_train_norm, upper_train_norm]).T
    ytr = data_shuffled['strength'].head(n_train).values

    #xts = np.array([pass_length_test, numeric_test, symbols_test, upper_test]).T
    xts = np.array([pass_length_test, numeric_test_norm, symbols_test_norm, upper_test_norm]).T
    yts = data_shuffled['strength'].tail(n_test).values

    return (xtr, ytr, xts, yts)
