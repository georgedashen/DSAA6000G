# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:37:47 2022

@author: 陈焯阳
"""

from pandas.core.indexing import is_label_like
from typing_extensions import final
from config import *
from toolkit import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import Normalizer
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor    


def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]


if __name__ == "__main__":
    #first let's merge back train and validation, denote their area(0-7) and flags(train/valid)
    final_list = []
    
    for i in range(8):
        for flag in ['train', 'valid']:
            res = pd.read_csv(f"./area_feature_raw/area_{i}_{flag}.csv", index_col=1)
            res.drop(['Unnamed: 0'],axis=1,inplace=True)
            res['area'] = i
            res['flag'] = flag
            final_list.append(res)
    
    final_res = pd.concat(final_list)            
    final_res.to_csv('./data/feature_processed.csv')
    
    filter_res = pd.read_csv("./data/feature_filtered.csv")
    
    for i in range(8):
        for flag in ['train','valid']:
            _final_res = filter_res[filter_res.area==i]
            _final_res = _final_res[filter_res.flag==flag]
            _final_res.drop(['area','flag'],axis=1,inplace=True)
            _final_res.to_csv(f'./area_feature/area_{i}_{flag}.csv')
