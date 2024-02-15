#!/usr/bin/env python
# coding: utf-8

import time
import math
import numpy as np
import pandas as pd
import edgeimpulse as ei

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Ftrl
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, SimpleRNN, GRU
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import roc_curve
from collections import Counter

from ClassificationPerformanceIndexes import classificationPerformanceIndexes, printClassificationPerformanceIndexes

# ## Edge Impulse

#Set up Edge Impulse for the Arduino Board
def EdgeImpulseProfile(model):
  ei.API_KEY = "ei_7af3f6dc06fe8069edbfa109b4c60bb1f623c82e5f16c557c66aaaa3adb042d5"
  labels = ["0", "1"]
  num_classes = len(labels)
  deploy_filename = "my_model_cpp.zip"
  try:
    profile = ei.model.profile(model=model, device='arduino-nano-33-ble')
    print(profile.summary())
  except Exception as e:
    print(f"Could not profile:{e}")

# ## K-fold split
def TrainingKfold (X, train, test):
    X_train = X.iloc[train,:X.shape[1]-1]
    y_train = X.iloc[train,-1:]
    X_test = X.iloc[test,:X.shape[1]-1]
    y_test = X.iloc[test,-1:]
    return X_train, y_train, X_test, y_test

# ## Classification with Neural Networks

# ### NN

def NNModel (size, dropout_percentage, metric):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(size,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_percentage))
    model.add(Dense(1, activation='sigmoid')
    model.compile(optimizer='Adam', loss="binary_crossentropy", metrics=metric)
    model.summary()
    return model

def NN_method (model, X_train, y_train, X_test, y_test, batch, epochs, results):
    print('Implementing NN...')
    start = time.time()
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
    #train model
    history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, callbacks = es, validation_data = (X_test,y_test), verbose = 0)
    #test model
    nn_ind = model.predict(X_test, batch_size = batch)
    end = time.time()
    t = round(end - start,2)
    #calculate evaluation values
    fpr, tpr, thrs =roc_curve(y_test, nn_ind)
    acc, snv, spc, tt = classificationPerformanceIndexes (y_test, np.reshape(nn_ind, nn_ind.shape[0]), t)
    results.loc['NN', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('NN', acc, snv, spc)
    print('NN finished in', t,'sec\n')
    #run edge impulse tests
    EdgeImpulseProfile(model)

def NN_method_Kfold(X, kf, cols, model, batch, epochs, results):
    results_list = pd.DataFrame(columns = cols)
    start = time.time()
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
        for train, test in kf.split(ft):
        #split and reshape the data
        X_train, y_train, X_test, y_test = TrainingKfold (ft, train, test)
        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
        #train model
        history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_test,y_test), callbacks = [es,checkpoint], verbose = 0)
        #test model
        nn_ind = (model.predict(X_test, batch_size = batch) >= 0.5).astype('int')
        #calculate evaluation values
        results_list.loc[results_list.shape[0], :] = classificationPerformanceIndexes (y_test, np.reshape(nn_ind, nn_ind.shape[0]), 0)
    end = time.time()
    t = round(end - start,2)
    #calculate the average performance
    acc, snv, spc, tt = np.array(results_list.mean(axis=0))
    results.loc['NN Kfold', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('NN Kfold', acc, snv, spc)
    print('NN finished in', t,'sec\n')

def CompleteNN (train_dat, test_dat, train_ind, test_ind, results, ft, kf, perfInd, epochs, batch, dropout_percentage, metric):
    X_train = train_dat
    y_train = train_ind.astype(int)
    X_test = test_dat
    y_test = test_ind.values.astype(int)

    nn_model = NNModel (train_dat.shape[1], dropout_percentage, metric)
    NN_method (nn_model, X_train, y_train, X_test, y_test, batch, epochs, results)
    NN_method_Kfold (ft, kf, perfInd, nn_model, batch, epochs, results)


# ### CNN

def CNNModel (size, dropout_percentage, metric):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(size, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_percentage))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer = 'Adam', loss = "binary_crossentropy", metrics = metric)
    return model

def CNN_method (model, X_train, y_train, X_test, y_test, batch, epochs, results):
    print('Implementing CNN...')
    start = time.time()
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
    #train model
    history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_test,y_test), callbacks = es, verbose = 0)
    #test model
    cnn_ind = model.predict(X_test, batch_size = batch)
    end = time.time()
    t = round(end - start, 2)
    #calculate evaluation values
    acc, snv, spc, tt = classificationPerformanceIndexes (y_test, np.reshape(cnn_ind, cnn_ind.shape[0]), t)
    results.loc['CNN', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('CNN', acc, snv, spc)
    print('CNN finished in', t,'sec\n')
    #run edge impulse tests
    EdgeImpulseProfile(model)

def CNN_method_Kfold(ft, kf, cols, model, batch, epochs, results):
    results_list = pd.DataFrame(columns = cols)
    print('Implementing CNN k-fold...')
    start = time.time()
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
    for train, test in kf.split(ft):
        #split and reshape the data
        X_train, y_train, X_test, y_test = TrainingKfold (ft, train, test)
        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
        #train model
        history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_test,y_test), callbacks = [es,checkpoint], verbose = 0)
        #test model
        cnn_ind = (model.predict(X_test, batch_size = batch) >= 0.5).astype('int')
        #calculate evaluation values
        results_list.loc[results_list.shape[0], :] = classificationPerformanceIndexes (y_test, np.reshape(cnn_ind, cnn_ind.shape[0]), 0)
    end = time.time()
    t = round(end - start,2)
    #calculate the average performance
    acc, snv, spc, tt = np.array(results_list.mean(axis=0))
    results.loc['CNN Kfold', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('CNN Kfold', acc, snv, spc)
    print('CNN finished in', t,'sec\n')

def CompleteCNN (train_dat, test_dat, train_ind, test_ind, results, ft, kf, perfInd, epochs, batch, dropout_percentage, metric):
    X_train = np.reshape(train_dat.values, (train_dat.shape[0], train_dat.shape[1], 1))
    y_train = train_ind.astype(int)
    X_test = np.reshape(test_dat.values, (test_dat.shape[0], test_dat.shape[1], 1))
    y_test = test_ind.values.astype(int)

    cnn_model = CNNModel (train_dat.shape[1], dropout_percentage, metric)
    CNN_method (cnn_model, X_train, y_train, X_test, y_test, batch, epochs, results)
    CNN_method_Kfold (ft, kf, perfInd, cnn_model, batch, epochs, results)

# ### RNN

def RNNModel (size, rnn_units, dense_units, dropout_percentage, metric):
    model = Sequential()
    model.add(SimpleRNN(rnn_units, unroll=True))
    model.add(Dropout(dropout_percentage))
    model.add(Dense(dense_units, activation = 'ReLU'))
    model.add(Dropout(dropout_percentage/2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'Adam', loss = "binary_crossentropy", metrics = metric)
    return model

def RNN_method (model, X_train, y_train, X_test, y_test, batch, epochs, results):
    print('Implementing RNN...')
    start = time.time()
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
    #train model
    history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_test,y_test), callbacks = es, verbose = 0)
    #test model
    rnn_ind = (model.predict(X_test, batch_size = batch) >= 0.5).astype('int')
    end = time.time()
    t = round(end - start,2)
    #calculate evaluation values
    acc, snv, spc, tt = classificationPerformanceIndexes (y_test, np.reshape(rnn_ind, rnn_ind.shape[0]), t)
    results.loc['RNN', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('RNN', acc, snv, spc)
    print('RNN finished in', t,'sec\n')
    #run edge impulse tests
    EdgeImpulseProfile(model)

def RNN_method_Kfold(X, kf, cols, model, batch, epochs, results):
    print('Implementing RNN k-fold...')
    results_list = pd.DataFrame(columns = cols)
    start = time.time()
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
    for train, test in kf.split(ft):
        #split and reshape the data
        X_train, y_train, X_test, y_test = TrainingKfold (ft, train, test)
        X_train = np.reshape(X_train.values, (X``_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
        #train model
        history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_test,y_test), callbacks = [es,checkpoint], verbose = 0)
        #test model
        rnn_ind = (model.predict(X_test, batch_size = batch) >= 0.5).astype('int')
        #calculate evaluation values
        results_list.loc[results_list.shape[0], :] = classificationPerformanceIndexes (y_test, np.reshape(rnn_ind, rnn_ind.shape[0]), 0)
    end = time.time()
    t = round(end - start,2)
    #calculate the average performance
    acc, snv, spc, tt = np.array(results_list.mean(axis=0))
    results.loc['RNN Kfold', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('RNN Kfold', acc, snv, spc)
    print('RNN finished in', t,'sec\n')

def CompleteRNN (train_dat, test_dat, train_ind, test_ind, results, ft, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric):
    X_train = np.reshape(train_dat.values, (train_dat.shape[0], 1, train_dat.shape[1]))
    y_train = train_ind.astype(int)
    X_test = np.reshape(test_dat.values, (test_dat.shape[0], 1, test_dat.shape[1]))
    y_test = test_ind.values.astype(int)

    rnn_model = RNNModel (train_dat.shape[1], rnn_units, dense_units, dropout_percentage, metric)
    RNN_method (rnn_model, X_train, y_train, X_test, y_test, batch, epochs, results)
    RNN_method_Kfold (ft, kf, perfInd, rnn_model, batch, epochs, results)

# ### LSTM

def LstmModel (size, rnn_units, dense_units, dropout_percentage, metric):
    model = Sequential()
    model.add(LSTM(rnn_units, unroll=True))
    model.add(Dropout(dropout_percentage))
    model.add(Dense(dense_units, activation = 'relu'))
    model.add(Dropout(dropout_percentage))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'Adam', loss = "binary_crossentropy", metrics = metric)
    return model

def LSTM_method (model, X_train, y_train, X_test, y_test, batch, epochs, results):
    print('Implementing LSTM...')
    start = time.time()   
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
    #train model
    history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_test,y_test), callbacks = es, verbose = 0)
    #test model
    lstm_ind = (model.predict(X_test, batch_size = batch) >= 0.5).astype('int')
    end = time.time()
    t = round(end - start,2)
    #calculate evaluation values
    acc, snv, spc, tt = classificationPerformanceIndexes (y_test, np.reshape(lstm_ind, lstm_ind.shape[0]), t)
    results.loc['LSTM', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('LSTM', acc, snv, spc)
    print('LSTM finished in', t,'sec\n')
    #run edge impulse tests
    EdgeImpulseProfile(model)

def LSTM_method_Kfold(X, kf, cols, model, batch, epochs, results):
    print('Implementing LSTM k-fold...')
    results_list = pd.DataFrame(columns = cols)
    start = time.time()
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
    for train, test in kf.split(ft):
        #split and reshape the data
        X_train, y_train, X_test, y_test = TrainingKfold (ft, train, test)
        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
        #train model
        history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_test,y_test), callbacks = [es,checkpoint], verbose = 0)
        #test model
        lstm_ind = (model.predict(X_test, batch_size = batch) >= 0.5).astype('int')
        #calculate evaluation values
        results_list.loc[results_list.shape[0], :] = classificationPerformanceIndexes (y_test, np.reshape(lstm_ind, lstm_ind.shape[0]), 0)
    end = time.time()
    t = round(end - start,2)
    #calculate the average performance
    acc, snv, spc, tt = np.array(results_list.mean(axis=0))
    results.loc['LSTM Kfold', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('LSTM Kfold', acc, snv, spc)
    print('LSTM finished in', t,'sec\n')

def CompleteLSTM (train_dat, test_dat, train_ind, test_ind, results, ft, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric):
    X_train = np.reshape(train_dat.values, (train_dat.shape[0], 1, train_dat.shape[1]))
    y_train = train_ind.astype(int)
    X_test = np.reshape(test_dat.values, (test_dat.shape[0], 1, test_dat.shape[1]))
    y_test = test_ind.values.astype(int)

    lstm_model = LstmModel (train_dat.shape[1], rnn_units, dense_units, dropout_percentage, metric)
    LSTM_method (lstm_model, X_train, y_train, X_test, y_test, batch, epochs, results)
    LSTM_method_Kfold (ft, kf, perfInd, lstm_model, batch, epochs, results)

# ### GRU

def GRUModel (size, rnn_units, dense_units, dropout_percentage, metric):
    model = Sequential()
    model.add(GRU(rnn_units, unroll=True))
    model.add(Dropout(dropout_percentage))
    model.add(Dense(dense_units, activation = 'relu'))
    model.add(Dropout(dropout_percentage/2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer = 'Adam', loss = "binary_crossentropy", metrics = metric)
    return model

def GRU_method (model, X_train, y_train, X_test, y_test, batch, epochs, results):
    print('Implementing GRU...')
    start = time.time()
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
    #train model
    history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_test,y_test), callbacks = es, verbose = 0)
    #test model
    gru_ind = (model.predict(X_test, batch_size = batch) >= 0.5).astype('int')
    end = time.time()
    t = round(end - start,2)
    #calculate evaluation values
    acc, snv, spc, tt = classificationPerformanceIndexes (y_test, np.reshape(gru_ind, gru_ind.shape[0]), t)
    results.loc['GRU', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('GRU', acc, snv, spc)
    print('GRU finished in', t,'sec\n')
    #run edge impulse tests
    EdgeImpulseProfile(model)

def GRU_method_Kfold(X, kf, cols, model, batch, epochs, results):
    print('Implementing GRU k-fold...')
    results_list = pd.DataFrame(columns = cols)
    start = time.time()
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, mode = 'auto', restore_best_weights = True, verbose = 0)
    for train, test in kf.split(ft):
        #Split and reshape the data
        X_train, y_train, X_test, y_test = TrainingKfold (ft, train, test)
        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
        #train model
        history = model.fit(X_train, y_train, batch_size = batch, epochs = epochs, validation_data = (X_test,y_test), callbacks = [es,checkpoint], verbose = 0)
        #test model
        gru_ind = (model.predict(X_test, batch_size = batch) >= 0.5).astype('int')
        #calculate evaluation values
        results_list.loc[results_list.shape[0], :] = classificationPerformanceIndexes (y_test, np.reshape(gru_ind, gru_ind.shape[0]), 0)
    end = time.time()
    t = round(end - start,2)
    #calculate the average performance
    acc, snv, spc, tt = np.array(results_list.mean(axis=0))
    results.loc['GRU Kfold', :] = acc, snv, spc, t
    printClassificationPerformanceIndexes('GRU Kfold', acc, snv, spc)
    print('GRU finished in', t,'sec\n')

def CompleteGRU (train_dat, test_dat, train_ind, test_ind, results, ft, kf, perfInd, epochs, batch, rnn_units, dense_units, dropout_percentage, metric):
    X_train = np.reshape(train_dat.values, (train_dat.shape[0], 1, train_dat.shape[1]))
    y_train = train_ind.astype(int)
    X_test = np.reshape(test_dat.values, (test_dat.shape[0], 1, test_dat.shape[1]))
    y_test = test_ind.values.astype(int)

    gru_model = GRUModel (train_dat.shape[1], rnn_units, dense_units, dropout_percentage, metric)
    GRU_method (gru_model, X_train, y_train, X_test, y_test, batch, epochs, results)
    GRU_method_Kfold (ft, kf, perfInd, gru_model, batch, epochs, results)
