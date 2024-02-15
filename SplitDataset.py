#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split

# ## Training-Test Datasets 

#Split dataset with 70-30 method
def createTrainingAndTestDatasets(dataset, test_ratio):
    X = dataset.loc[:, dataset.columns != 'seizure']
    y = dataset['seizure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, shuffle = True, random_state=42)
    return X_train, X_test, y_train, y_test
