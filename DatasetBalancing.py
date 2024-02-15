#!/usr/bin/env python
# coding: utf-8

from imblearn.under_sampling import ClusterCentroids

# ### Undersampling majority class

#ClusterCentroids with rate 20%
def undersamplingClusterCentroids(ft, ft_index):
    cc = ClusterCentroids(sampling_strategy = 0.2)
    cc_features, cc_indicator = cc.fit_resample(ft, ft_index)
    cc_features['seizure'] = cc_indicator
    return cc_features

def majorityUndersampling (ft, ft_index):
        return undersamplingClusterCentroids(ft, ft_index)
