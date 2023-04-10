import numpy as np

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, (Y.max()+1)))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y
    return one_hot_Y

def shuffle(X, Y):
    idx = [i for i in range(X.shape[0])]
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    return X, Y