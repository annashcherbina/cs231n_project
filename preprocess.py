#This module performs mean subtraction and normalization
import numpy as np 

def mean_subtract(X):
    X=np.asarray(X) 
    X -= (np.mean(X, axis = 0))
    return X

def normalize(X):
    X=(np.asarray(X)).astype(np.float64) 
    X /= np.std(X, axis = 0)
    return X

def preprocess(X_train,X_valid,X_test):
    mean_train=np.mean(X_train,axis=0)
    std_train=np.std(X_train,axis=0) 
    X_train=(X_train-mean_train)/std_train 
    X_valid=(X_valid-mean_train)/std_train 
    X_test=(X_test-mean_train)/std_train 

    #X_train=normalize(mean_subtract(X_train))
    #X_valid=normalize(mean_subtract(X_valid))
    #X_test=normalize(mean_subtract(X_test))
    return X_train,X_valid,X_test

