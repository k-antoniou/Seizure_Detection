#!/usr/bin/env python
# coding: utf-8

from os import ftruncate
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# ## Feature Selection

#Select k best features using k best method
def featureSelection(df, kbest):
    ft = df.iloc[:, :df.columns.size-1]
    selector = SelectKBest(f_classif, k=kbest)
    selector.fit(ft, df['seizure'])
    cols_idxs = selector.get_support(indices=True)
    new_ft = ft.iloc[:,cols_idxs]
    new_ft = new_ft.join(df['seizure'])
    return new_ft, cols_idxs
    