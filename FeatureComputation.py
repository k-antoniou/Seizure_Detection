#!/usr/bin/env python
# coding: utf-8

import pywt
import math
import numpy as np
import pandas as pd
from pyentrp import entropy
from tqdm.notebook import tqdm
from scipy.stats import skew, kurtosis, variation

# ## Feature Extraction

# ### Time Domain Features

'''
https://stackoverflow.com/questions/30272538/python-code-for-counting-number-of-zero-crossings-in-an-array
https://stackoverflow.com/questions/5613244/root-mean-square-in-numpy-and-complications-of-matrix-and-arrays-of-numpy
'''
def computeTimeDomainFeatures (x):
    mean = np.mean(x)
    var = np.var(x)
    sk = skew(x)
    kurt = kurtosis(x)
    std = np.std(x)
    median = np.median(x)
    zcr = ((x[:-1] * x[1:]) < 0).sum() / len(x)
    if x.mean() != 0:
        cv = variation(x)
    else:
        cv = math.nan
    if x.size > 0:
        rms = np.sqrt(x.dot(x)/x.size)
    else:
        rms = math.nan
    p2p = x.max() - x.min()
    sampEn = entropy.sample_entropy(x, 1)[0]
    return mean, var, sk, kurt, std, median, zcr, cv, rms, p2p, sampEn

# ### Feature computation

#Compute features based on a 2s window
def featureExtraction (df, sample_rate, step):
    print('Feature Extraction')
    ft = pd.DataFrame()
    c = 0
    for i in tqdm(range (0, df.shape[0], step)):
        temp = df.iloc[i:i+step]
        for j in range(0, df.shape[1]-1):
            s = np.array(temp.iloc[:, j])
            # Time Domain Features
            ft.loc[c, 'mean'+str(j)], ft.loc[c, 'var'+str(j)], ft.loc[c, 'skew'+str(j)],ft.loc[c, 'kurt'+str(j)], ft.loc[c, 'std'+str(j)], ft.loc[c, 'median'+str(j)], ft.loc[c, 'zcr'+str(j)], ft.loc[c, 'cv'+str(j)], ft.loc[c, 'rms'+str(j)], ft.loc[c, 'p2p'+str(j)],ft.loc[c, 'sampEn'+str(j)] = computeTimeDomainFeatures(s)
        ft.loc[c, 'seizure'] = temp['seizure'].value_counts().idxmax()
        c = c + 1
    return ft

