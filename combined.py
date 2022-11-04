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

binary_model = 'precision_0_Eval100'
magnitude_model = 'weightedPrecisionEval1000' #best is weightedPrecisionEval1000

if __name__ == "__main__":
    Correct = Total = TP = TN = FP = FN =0
    
    overall_M = pd.Series()
    overall_pred = pd.Series()
    
    for area in (0,1,2,3,4,5,6,7):
        print(f"\n\nArea_{area}:")
        valid_data = get_area_data(area, 'valid')
        valid_M = valid_data['label_M']
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
        
        result_combined = result_mag
        for i, ss in enumerate(result_bin):
            if(ss==1):
                result_combined[i] = 0
        
        #result_combined = np.array(result_mag.reshape(39,)) * np.array(result_bin)
        #result_combined = result_combined.reshape(39,)
        
        cpf = open(str(area)+'_weighted.txt','w+')
        for i in range(len(result_combined)):
            print(f"{i}, pre:{result_combined[i]}, origin:{valid_M[i]}", file=cpf)
        
        correct = (result_combined==valid_M).sum()
        total = len(result_combined)
        tp = ((result_combined!=0) * (valid_M!=0)).sum()
        tn = ((result_combined==0) * (valid_M==0)).sum()
        fp = ((result_combined!=0) * (valid_M==0)).sum()
        fn = ((result_combined==0) * (valid_M!=0)).sum()
        accuracy = correct / total
        recall = tp / (tp + fn) if (tp + fn)!=0 else 0
        precision = tp / (tp + fp) if (tp + fp)!=0 else 0
        f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn)!=0 else 0
        
        Correct += correct
        Total += total
        TP += tp
        TN += tn
        FP += fp
        FN += fn
        overall_M = overall_M.append(valid_M)
        overall_pred = overall_pred.append(result_combined)
        
        print(f"accuracy: {accuracy}, {correct}/{total}")
        print(f"recall: {recall}, {tp}/({tp}+{fn})")
        print(f"precision: {precision}, {tp}/({tp}+{fp})")
        print(f"f1: {f1}")
        print(classification_report(valid_M, result_combined, labels=range(0,5)))
        print(f"accuracy: {accuracy}, {correct}/{total}", file=cpf)
        print(f"recall: {recall}, {tp}/({tp}+{fn})", file=cpf)
        print(f"precision: {precision}, {tp}/({tp}+{fp})", file=cpf)
        print(f"f1: {f1}", file=cpf)
        print(classification_report(valid_M, result_combined, labels=range(0,5)), file=cpf)
        cpf.close()

    cpf = open('overall_weighted.txt','w+')
    Accuracy = Correct / Total
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1 = 2*TP / (2*TP+FP+FN)
    
    print("\n\nOverall Performance")
    print(f"Accuracy: {Accuracy}, {Correct}/{Total}")
    print(f"Recall: {Recall}, {TP}/({TP}+{FN})")
    print(f"Precision: {Precision}, {TP}/({TP}+{FP})")
    print(f"F1: {F1}")
    print(classification_report(overall_M, overall_pred, labels=range(0,5)))
   
    print("Overall Performance", file=cpf)
    print(f"Accuracy: {Accuracy}, {Correct}/{Total}", file=cpf)
    print(f"Recall: {Recall}, {TP}/({TP}+{FN})", file=cpf)
    print(f"Precision: {Precision}, {TP}/({TP}+{FP})", file=cpf)
    print(f"F1: {F1}", file=cpf)
       
    print(classification_report(overall_M, overall_pred, labels=range(0,5)), file=cpf)
    frame = { 'pred': overall_pred, 'truth': overall_M }
    df = pd.DataFrame(frame)
    print(pd.crosstab(df['truth'],df['pred']))
    print(pd.crosstab(df['truth'],df['pred']), file=cpf)
    
    cpf.close()