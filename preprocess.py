#This module performs mean subtraction and normalization
import numpy as np 

def mean_subtract(X):
    X=np.asarray(X) 
    X -= np.mean(X, axis = 0)
    return X

def normalize(X):
    X=np.asarray(X) 
    X /= np.std(X, axis = 0)
    return X

def preprocess(X_train,X_valid,X_test):
    X_train=normalize(mean_subtract(X_train))
    X_valid=normalize(mean_subtract(X_valid))
    X_test=normalize(mean_subtract(X_test))
    return X_train,X_valid,X_test

