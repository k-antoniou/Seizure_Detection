#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ## Feature Normalization

def featureNormalization(ft):
    scaled_df = StandardScaler().fit_transform(ft.iloc[:, :ft.shape[1]-1])
    norm_ft = pd.DataFrame(scaled_df)
    norm_ft['seizure'] = ft['seizure'].copy()
    return norm_ft

# ## Remove NAN values

def removeNonNumericValues(df):
    df.replace([np.inf, -np.inf], np.nan, inplace = True)
    df.dropna(inplace = True)
