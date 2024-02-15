#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from FeatureSelection import  featureSelection
from SplitDataset import createTrainingAndTestDatasets
from DatasetBalancing import minorityOversampling, majorityUndersampling
from FeatureNormalization import featureNormalization, removeNonNumericValues
from ClassificationPerformanceIndexes import classificationPerformanceIndexes, printClassificationPerformanceIndexes
from ClassificationModels import CompleteNN, CompleteCNN, CompleteRNN, CompleteLSTM, CompleteGRU

# ### Feature Extraction

def featureExtraction (df):
    #Normalization
    removeNonNumericValues(df)
    ft = featureNormalization(df)
    print('Normalized features')
    removeNonNumericValues(ft)
    #Undersampling
    size = df['seizure'].value_counts()
    print('Undersampling the majority class using ClusterCentroids')
    ft = majorityUndersampling(df.loc[:, df.columns != 'seizure'], df['seizure'], train)
    removeNonNumericValues(ft)
    print('Majority class downsampled from (', size[0], ', ', ft.shape[1], ') to ', ft.shape, sep = '')
    return ft

# ### Create Train and Test Sets

def trainTestData (features, test_ratio, k_fold, perfInd):
    X_train, X_test, y_train, y_test = createTrainingAndTestDatasets(features, test_ratio)
    results = pd.DataFrame(columns = perfInd)
    kf = KFold(n_splits = k_fold, shuffle = True, random_state=42)
    return X_train, X_test, y_train, y_test, results, kf

# ### Experiment

def experiment (df, test_ratio, k_fold, epochs, batch, dropout_percentage, metric, perfInd):
    #Preprocessing
    df = featureExtraction(df)
    
    #Feature Seletion
    print('Feature Selection')
    df_90, cols_90 =  featureSelection(df, 90)
    df_45, cols_45 =  featureSelection(df, 45)
    df_25, cols_25 =  featureSelection(df, 25)
    df_12, cols_12 =  featureSelection(df, 12)
    
    #Train test split
    X_train, X_test, y_train, y_test, results, kf = trainTestData (df,test_ratio, k_fold, perfInd)
    
    #Set-up for rnns
    rnn_units = 64
    dense_units = 16
    
    #Run for all features
    CompleteNN(X_train, X_test, y_train, y_test, results, df, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteCNN(X_train, X_test, y_train, y_test, results, df, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteRNN(X_train, X_test, y_train, y_test, results, df, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteLSTM(X_train, X_test, y_train, y_test, results, df, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteGRU(X_train, X_test, y_train, y_test, results, df, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    
    #Run for 90 features
    X_train_90=X_train.iloc[:,cols_90]
    X_test_90=X_test.iloc[:,cols_90]
    
    CompleteNN(X_train_90, X_test_90, y_train, y_test, results, df_90, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteCNN(X_train_90, X_test_90, y_train, y_test, results, df_90, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteRNN(X_train_90, X_test_90, y_train, y_test, results, df_90, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteLSTM(X_train_90, X_test_90, y_train, y_test, results, df_90, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteGRU(X_train_90, X_test_90, y_train, y_test, results, df_90, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    
    #Run for 45 features
    X_train_45=X_train.iloc[:,cols_45]
    X_test_45=X_test.iloc[:,cols_45]

    CompleteNN(X_train_45, X_test_45, y_train, y_test, results, df_45, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteCNN(X_train_45, X_test_45, y_train, y_test, results, df_45, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteRNN(X_train_45, X_test_45, y_train, y_test, results, df_45, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteLSTM(X_train_45, X_test_45, y_train, y_test, results, df_45, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteGRU(X_train_45, X_test_45, y_train, y_test, results, df_45, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    
    #Run for 25 features
    X_train_25=X_train.iloc[:,cols_25]
    X_test_25=X_test.iloc[:,cols_25]

    CompleteNN(X_train_25, X_test_25, y_train, y_test, results, df_25, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteCNN(X_train_25, X_test_25, y_train, y_test, results, df_25, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteRNN(X_train_25, X_test_25, y_train, y_test, results, df_25, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteLSTM(X_train_25, X_test_25, y_train, y_test, results, df_25, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteGRU(X_train_25, X_test_25, y_train, y_test, results, df_25, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    #Run for 12 features
    X_train_12=X_train.iloc[:,cols_12]
    X_test_12=X_test.iloc[:,cols_12]

    CompleteNN(X_train_12, X_test_12, y_train, y_test, results, df_12, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteCNN(X_train_12, X_test_12, y_train, y_test, results, df_12, kf, perfInd, epochs, batch, dropout_percentage, metric)
    CompleteRNN(X_train_12, X_test_12, y_train, y_test, results, df_12, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteLSTM(X_train_12, X_test_12, y_train, y_test, results, df_12, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    CompleteGRU(X_train_12, X_test_12, y_train, y_test, results, df_12, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric)
    
    return results
