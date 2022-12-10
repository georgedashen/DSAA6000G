import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report
import os 
import re

from hyperopt import hp, fmin, tpe, Trials, space_eval

import warnings
warnings.filterwarnings('ignore')

def get_area_data(area_id=0, flag='train', path='./area_feature/'):
    file_name = path+'area_' + str(area_id) + '_' + flag + '.csv'
    return pd.read_csv(file_name,index_col=0)

MODEL_NAME = "precision_0_Eval200"
def optimize(config):
    params.update(config)
    model = lgb.train(params, trn_data, valid_sets=[trn_data,val_data],verbose_eval=500,
                    early_stopping_rounds=10)
    oof_lgb = model.predict(valid_feature, num_iteration=model.best_iteration)
    pred = pd.Series([1 if i > 0.5 else 0 for i in oof_lgb])
    correct = (pred==valid_M).sum()
    total = len(oof_lgb)
    tp = ((pred!=0) * (valid_M!=0)).sum()
    tn = ((pred==0) * (valid_M==0)).sum()
    fp = ((pred!=0) * (valid_M==0)).sum()
    fn = ((pred==0) * (valid_M!=0)).sum()
    accuracy = correct / total
    recall = tp / (tp + fn) if (tp + fn)!=0 else 0 #use this for binary_hp
    precision = tp / (tp + fp) if (tp + fp)!=0 else 0
    #precision0 = tn / (tn + fn) if (tp + fp)!=0 else 0
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn)!=0 else 0
    #auc = roc_auc_score(np.array(valid_M), np.array(pred))
    
    return -precision

MAX_EVALS = 200

if __name__ == "__main__":
    Correct = Total = TP = TN = FP = FN =0
    
    printOverall = True
    if(printOverall): 
        overall = open(f'./evaluation/overall_binary_hp_{MODEL_NAME}.txt','w+')
        overall_M = pd.Series()
        overall_pred = pd.Series()
    
    for area in (0,1,2,3,4,5,6,7): #train false positive area
        print(f"Area_{area}:")
        train_data = get_area_data(area, 'train')
        valid_data = get_area_data(area, 'valid')

    #训练数据
        print("Preprocessing......")
        #数据平衡，升采样
        long_data = train_data[train_data['label_M']==0]
        short_data = train_data[train_data['label_M']!=0]
        if len(long_data)<len(short_data) :
            long_data,short_data = short_data,long_data
        short_data = short_data.sample(len(long_data), replace=True)
        train_data = pd.concat([long_data,short_data])
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        del long_data
        del short_data
        
        train_data.reset_index(drop=True, inplace=True)
        valid_data.reset_index(drop=True, inplace=True)

        target_M = train_data['label_M']
        train_feature = train_data.drop(['label_M','label_long','label_lati','Day'],axis=1)

        #震级类别化
        for i, ss in enumerate(target_M):
            if(ss<3.5):
                target_M.iloc[i] = 1
            else:
                target_M.iloc[i] = 0

        #震级赋权
        train_data['weight'] = None
        train_data['label_M'] = target_M
        train_data['weight'][train_data['label_M']==0] = 1
        train_data['weight'][train_data['label_M']==1] = 1
        weight_T = train_data['weight'].values
    #验证数据
        valid_M = valid_data['label_M']
        valid_feature = valid_data.drop(['label_M','label_long','label_lati','Day'],axis=1)
        for i, ss in enumerate(valid_M):
            if(ss<3.5):
                valid_M.iloc[i] = 1
            else:
                valid_M.iloc[i] = 0

        valid_data['weight'] = None
        valid_data['label_M'] = valid_M
        valid_data['weight'][valid_data['label_M']==0] = 1
        valid_data['weight'][valid_data['label_M']==1] = 1
        weight_V = valid_data['weight'].values
        
        trn_data = lgb.Dataset(train_feature, label=target_M,weight=weight_T)
        val_data = lgb.Dataset(valid_feature, label=valid_M, weight=weight_V)
        
        print("\nSearching parameters......")
        
        params = {
            'num_iterations': 1000,
            'num_leaves': 10,
            'learning_rate': 0.05,
            "boosting": "gbdt",
            'objective': 'binary', #转为分类问题
            #'num_class': 5,
            # 'objective': 'regression',
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 2,
            "lambda_l1": 0.05,
            "lambda_l2": 0.05,
            "nthread": 6,
            'feature_pre_filter': False,
            'min_child_samples': 10,
            'max_bin': 200,
            'verbose' : -1
        }
        
        spaces = {
            "learning_rate":hp.loguniform("learning_rate",np.log(0.001),np.log(0.5)),
            "num_iterations": hp.choice("num_iterations",range(100,500)),
            "boosting": hp.choice("boosting", ["gbdt","rf"]),
            "num_leaves":hp.choice("num_leaves",range(2,43)),
            "feature_fraction":hp.uniform("feature_fraction",0.5,1.0),
            "bagging_fraction":hp.uniform("bagging_fraction",0.5,1.0),
            "bagging_freq":hp.choice("bagging_freq",range(2,10)),
            "min_child_samples":hp.choice("min_child_samples", range(10,50)),
        }
        
        #optimization
        best = fmin(fn=optimize, 
                    space=spaces, 
                    algo=tpe.suggest, 
                    max_evals=MAX_EVALS,
                    trials=Trials())
        best_params = space_eval(spaces, best)
        
        #train
        print("\nTraining......")
        results = {}
        params.update(best_params)
        Bin = lgb.train(params, trn_data, valid_sets=[trn_data,val_data],
                        verbose_eval=500, early_stopping_rounds=10, evals_result=results)
        

        #评估
        oof_lgb = Bin.predict(valid_feature, num_iteration=Bin.best_iteration)
        pred = pd.Series([1 if i > 0.5 else 0 for i in oof_lgb])
        cpf = open('./evaluation/'+str(area)+'_binary_hp_recall.txt','w+')
        print(f'\nArea_{area}_validation:', file=overall)
        for i in range(len(pred)):
            print(f"{i}, pre:{pred[i]}, origin:{valid_M[i]}", file=cpf)
        
        correct = (pred==valid_M).sum()
        total = len(oof_lgb)
        tp = ((pred!=0) * (valid_M!=0)).sum()
        tn = ((pred==0) * (valid_M==0)).sum()
        fp = ((pred!=0) * (valid_M==0)).sum()
        fn = ((pred==0) * (valid_M!=0)).sum()
        accuracy = correct / total
        sensitivity = tp / (tp + fn) if (tp + fn)!=0 else 0
        specificity = tp / (tp + fp) if (tp + fp)!=0 else 0
        f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn)!=0 else 0
        Correct += correct
        Total += total
        TP += tp
        TN += tn
        FP += fp
        FN += fn
        print(f"\n\naccuracy: {accuracy}, {correct}/{total}")
        print(f"sensitivity(recall): {sensitivity}, {tp}/({tp}+{fn})")
        print(f"specificity(presicion): {specificity}, {tp}/({tp}+{fp})")
        print(f"f1: {f1}")
        print(f"\n\naccuracy: {accuracy}, {correct}/{total}", file=cpf)
        print(f"sensitivity(recall): {sensitivity}, {tp}/({tp}+{fn})", file=cpf)
        print(f"specificity(precision): {specificity}, {tp}/({tp}+{fp})", file=cpf)
        print(f"f1: {f1}", file=cpf)
        print(classification_report(valid_M, pred), file=cpf)
        cpf.close()
        
        if(printOverall):
            print(classification_report(valid_M, pred), file=overall)
            overall_M = overall_M.append(valid_M)
            overall_pred = overall_pred.append(pred)
        
        #保存区域模型
        Bin.save_model('./model/'+str(area)+f'_binary_hp_{MODEL_NAME}_model.txt')


    Accuracy = Correct / Total
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1 = 2*TP / (2*TP+FP+FN)
    print(f"\n\nAccuracy: {Accuracy}, {Correct}/{Total}")
    print(f"Recall: {Recall}, {TP}/({TP}+{FN})")
    print(f"Precision: {Precision}, {TP}/({TP}+{FP})")
    print(f"F1: {F1}")
    if(printOverall):
        print(f"\n\nAccuracy: {Accuracy}, {Correct}/{Total}", file=overall)
        print(f"recall: {Recall}, {TP}/({TP}+{FN})", file=overall)
        print(f"precision: {Precision}, {TP}/({TP}+{FP})", file=overall)
        print(f"F1: {F1}", file=overall)
        print(classification_report(overall_M, overall_pred))
        print(classification_report(overall_M, overall_pred), file=overall)
        overall.close()

# print("Evaluating model......")

# y_pred_train = gbm.predict(dftrain.drop('label',axis = 1), num_iteration=gbm.best_iteration)
# y_pred_test = gbm.predict(dftest.drop('label',axis = 1), num_iteration=gbm.best_iteration)

# train_score = f1_score(dftrain['label'],y_pred_train>0.5)
# val_score = f1_score(dftest['label'],y_pred_test>0.5)

# print('train f1_score: {:.5} '.format(train_score))
# print('valid f1_score: {:.5} \n'.format(val_score))

# fig2,ax2 = plt.subplots(figsize=(6,3.7),dpi=144)
# fig3,ax3 = plt.subplots(figsize=(6,3.7),dpi=144)
# lgb.plot_metric(results,ax = ax2)
# lgb.plot_importance(gbm,importance_type = "gain",ax=ax3)