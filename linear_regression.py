# train the model on a linear regression
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import rdkit
import random
from sklearn import linear_model

def data_split(X, y):
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=4)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.25, random_state=4)

    return X_train, X_test, y_train, y_test, X_validation, y_validation


def fit_linear_regression(X, y, lmbda=0.0, regularization=None):
    """
    Fit a ridge regression model to the data, with regularization parameter lmbda and a given
    regularization method.
    If the selected regularization method is None, fit a linear regression model without a regularizer.

    Does not fit the intersept in all cases.

    y = wx+c

    X: 2D numpy array of shape (n_samples, n_features corresponding to the fingerprint of the molecule)
    y: 1D numpy array of shape (n_samples,)
    lmbda: float, regularization parameter
    regularization: string, 'ridge' or 'lasso' or None

    Returns: The coefficients and intercept of the fitted model.
    """

    # choosing the right linear model:

    if regularization=='lasso':
        linreg=linear_model.Lasso(alpha=lmbda,fit_intercept=False)
    elif regularization==None:
        linreg=linear_model.LinearRegression(fit_intercept=False)
    else:
        linreg=linear_model.Ridge(alpha=lmbda, fit_intercept=False)
    
    # fitting the model to our training set

    linreg.fit(X,y)
    w = linreg.coef_ # coefficients
    c = linreg.intercept_ # intercept

    return w, c

def predict(X, w, c):
    """
    Return a linear model prediction for the data X.

    X: 2D numpy array of shape (n_samples, n_features) data
    w: 1D numpy array of shape (n_features,) coefficients
    c: float intercept

    Returns: 1D numpy array of shape (n_samples,)
    """
    # TODO

    # we predict the output of the regression, we could have also used the predict attribute of the linear models

    y_pred = X @ w +c
    return y_pred