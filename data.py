import numpy as np


def feature_extractor(data, n_train, n_test):
    """
    Extracts and normalizes password features for training and testing from a given dataset.
    Parameters:
        data (pandas.DataFrame): The input DataFrame containing at least 'password' and 'strength' columns.
        n_train (int): Number of samples to use for the training set.
        n_test (int): Number of samples to use for the test set.
    Returns:
        tuple: A tuple containing:
            - xtr (np.ndarray): Training feature matrix of shape (n_train, 4) with columns:
                [password length, normalized digit count, normalized symbol count, normalized uppercase count].
            - ytr (np.ndarray): Training target array of shape (n_train,).
            - xts (np.ndarray): Test feature matrix of shape (n_test, 4) with the same columns as xtr.
            - yts (np.ndarray): Test target array of shape (n_test,).
    Notes:
        - The dataset is shuffled before splitting.
        - Feature normalization is performed by dividing counts by password length (with division by zero avoided).
    """
    # Shuffle the dataset
    data_shuffled = data.sample(frac=1)  

    data_shuffled['password'] = data_shuffled['password'].astype(str)

    # Generate features
    pass_length_train = data_shuffled['password'].str.len().head(n_train).values
    digits_train = data_shuffled['password'].head(n_train).apply(
        lambda x: len([c for c in x if c.isdigit()])).values
    symbols_train = data_shuffled['password'].head(n_train).apply(
        lambda x: len([c for c in x if not c.isalnum()])).values
    upper_train = data_shuffled['password'].head(n_train).apply(
        lambda x: len([c for c in x if c.isupper()])).values
    
    # Normalize features by password length (avoid division by zero)
    digits_train_norm = digits_train / np.where(pass_length_train == 0, 1, pass_length_train)
    symbols_train_norm = symbols_train / np.where(pass_length_train == 0, 1, pass_length_train)
    upper_train_norm = upper_train / np.where(pass_length_train == 0, 1, pass_length_train)
    
    pass_length_test = data_shuffled['password'].str.len().tail(n_test).values
    digits_test = data_shuffled['password'].tail(n_test).apply(
        lambda x: len([c for c in x if c.isdigit()])).values
    symbols_test = data_shuffled['password'].tail(n_test).apply(
        lambda x: len([c for c in x if not c.isalnum()])).values
    upper_test = data_shuffled['password'].tail(n_test).apply(
        lambda x: len([c for c in x if c.isupper()])).values
    
    digits_test_norm = digits_test / np.where(pass_length_test == 0, 1, pass_length_test)
    symbols_test_norm = symbols_test / np.where(pass_length_test == 0, 1, pass_length_test)
    upper_test_norm = upper_test / np.where(pass_length_test == 0, 1, pass_length_test)
    
    # Split the dataset into features and target variable
    #xtr = np.array([pass_length_train, digits_train, symbols_train, upper_train]).T
    xtr = np.array([pass_length_train, digits_train_norm, symbols_train_norm, upper_train_norm]).T
    ytr = data_shuffled['strength'].head(n_train).values

    #xts = np.array([pass_length_test, digits_test, symbols_test, upper_test]).T
    xts = np.array([pass_length_test, digits_test_norm, symbols_test_norm, upper_test_norm]).T
    yts = data_shuffled['strength'].tail(n_test).values

    return (xtr, ytr, xts, yts)
