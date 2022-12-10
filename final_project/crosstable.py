# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:50:58 2022

@author: 陈焯阳
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report
import os 
import re
import warnings
warnings.filterwarnings('ignore')


def get_area_data(area_id=0, flag='train', path='./area_feature/'):
    file_name = path+'area_' + str(area_id) + '_' + flag + '.csv'
    return pd.read_csv(file_name,index_col=0)

binary_model = 'precision_0_Eval200'
magnitude_model = 'weightedPrecisionEval1000' #best is weightedPrecisionEval1000

if __name__ == "__main__":    
    overall_M_bin = pd.Series()
    overall_M = pd.Series()
    overall_pred_bin = pd.Series()
    overall_pred = pd.Series()
    
    for area in (0,1,2,3,4,5,6,7):
        print(f"\n\nArea_{area}:")
        valid_data = get_area_data(area, 'valid')
        valid_data.reset_index(drop=True, inplace=True)
        
        valid_M = valid_data['label_M'].copy()
        valid_M_bin = valid_data['label_M'].copy()
        valid_feature = valid_data.drop(['label_M','label_long','label_lati','Day'],axis=1)
        
        for i, ss in enumerate(valid_M):
            if(ss<3.5):
                valid_M.iloc[i] = 0
            elif(ss<4.0):
                valid_M.iloc[i] = 1
            elif(ss<4.5):
                valid_M.iloc[i] = 2
            elif(ss<5.0):
                valid_M.iloc[i] = 3
            else:
                valid_M.iloc[i] = 4
                                
        for i, ss in enumerate(valid_M_bin):
            if(ss<3.5):
                valid_M_bin.iloc[i] = 1
            else:
                valid_M_bin.iloc[i] = 0
        
        valid_data['weight'] = None
        valid_data['label_M'] = valid_M
        valid_data['weight'][valid_data['label_M']==0] = 1
        valid_data['weight'][valid_data['label_M']==1] = 1
        valid_data['weight'][valid_data['label_M']==2] = 1
        valid_data['weight'][valid_data['label_M']==3] = 1
        valid_data['weight'][valid_data['label_M']==4] = 1
        weight_V = valid_data['weight'].values
        
        Bin = lgb.Booster(model_file=f'./model/{area}_binary_hp_{binary_model}_model.txt')
        clf = lgb.Booster(model_file=f'./model/{area}_mag_{magnitude_model}_model.txt')
        
        result_bin = Bin.predict(valid_feature, num_iteration=Bin.best_iteration)
        result_bin = pd.Series([1 if i > 0.5 else 0 for i in result_bin])
        result_mag = np.matrix(clf.predict(valid_feature, num_iteration=clf.best_iteration))
        result_mag = pd.Series(np.array(result_mag.argmax(axis=1)).reshape(39,))
        
        
        overall_M_bin = overall_M_bin.append(valid_M_bin)
        overall_M = overall_M.append(valid_M)
        overall_pred_bin = overall_pred_bin.append(result_bin)
        overall_pred = overall_pred.append(result_mag)
        

    cpf = open('./evaluation/overall_combined_crosstable.txt','w+')  
    print('magnitude class:', file=cpf)
    print(classification_report(overall_M, overall_pred, labels=range(0,5)), file=cpf)
    frame = { 'pred': overall_pred, 'truth': overall_M }
    df = pd.DataFrame(frame)
    print(pd.crosstab(df['truth'],df['pred']))
    print(pd.crosstab(df['truth'],df['pred']), file=cpf)
    
    print('binary class:', file=cpf)
    print(classification_report(overall_M_bin, overall_pred_bin, labels=range(0,5)), file=cpf)
    frame = { 'pred': overall_pred_bin, 'truth': overall_M_bin }
    df = pd.DataFrame(frame)
    print(pd.crosstab(df['truth'],df['pred']))
    print(pd.crosstab(df['truth'],df['pred']), file=cpf)
    
    cpf.close()