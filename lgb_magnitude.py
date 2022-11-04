import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,log_loss,classification_report
import os 
import re

from hyperopt import hp, fmin, tpe, Trials, space_eval

import warnings
warnings.filterwarnings('ignore')

def get_area_data(area_id=0, flag='train', path='./area_feature/'):
    file_name = path+'area_' + str(area_id) + '_' + flag + '.csv'
    return pd.read_csv(file_name,index_col=0)


MODEL_NAME = "weightedPrecisionEval1000"  
def optimize(config):
    params.update(config)
    model = lgb.train(params, trn_data, valid_sets=[trn_data,val_data],verbose_eval=500,
                    early_stopping_rounds=10)
    
    pred = model.predict(valid_feature, num_iteration=model.best_iteration)
    acc = accuracy_score(pred.argmax(axis=1),valid_M)
    oof_lgb = np.matrix(model.predict(valid_feature, num_iteration=model.best_iteration))
    
    f1_array = []
    recall_array = []
    precision_array = []
    for i in range(0,5):
        tp = (np.array(oof_lgb.argmax(axis=1)!=i) * np.array((np.matrix(valid_M).T)!=i)).sum()
        tn = (np.array(oof_lgb.argmax(axis=1)==i) * np.array((np.matrix(valid_M).T)==i)).sum()
        fp = (np.array(oof_lgb.argmax(axis=1)!=i) * np.array((np.matrix(valid_M).T)==i)).sum()
        fn = (np.array(oof_lgb.argmax(axis=1)==i) * np.array((np.matrix(valid_M).T)!=i)).sum()
        f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn)!=0 else 0
        recall = tp / (tp + fn) if (tp + fn)!=0 else 0
        precision = tp / (tp + fp) if (tp + fp)!=0 else 0
        f1_array.append(f1)
        recall_array.append(recall)
        precision_array.append(precision)
    
    #auc = roc_auc_score(valid_M, clf.predict(valid_feature, num_iteration=clf.best_iteration), multi_class='ovr', labels=range(0,5))
    #logloss = log_loss(valid_M, model.predict(valid_feature),labels=range(0,5))
    #for multi-class, choose f1, logloss or accuracy
    #but given that most of cases in 0, choosing accuracy and logloss will force all prediction in 0
    # a weighted f1 score is preferred
    weighted_f1 = 0.125*f1_array[0] + 0.198*f1_array[1] + 0.229*f1_array[2] + 0.226*f1_array[3] + 0.222*f1_array[4]
    #weighted_f1 = sum(f1_array) *100
    #weighted_recall = 0.125*recall_array[0] + 0.198*recall_array[1] + 0.229*recall_array[2] + 0.226*recall_array[3] + 0.222*recall_array[4]
    #weighted_f1 = 0.019*f1_array[0] + 0.241*f1_array[1] + 0.245*f1_array[2] + 0.247*f1_array[3] + 0.248*f1_array[4]
    #weighted_recall = 0.019*recall_array[0] + 0.241*recall_array[1] + 0.245*recall_array[2] + 0.247*recall_array[3] + 0.248*recall_array[4]
    weighted_precision = 0.1*precision_array[0] + 5*precision_array[1] + 20*precision_array[2] + 10*precision_array[3] + 20*precision_array[4]
    #weighted_recall = 0.1*recall_array[1] + 0.2*recall_array[2] + 0.3*recall_array[3] + 0.4*recall_array[4]
    #weighted_recall = 0.33*recall_array[2] + 0.33*recall_array[3] + 0.33*recall_array[4]
    sum_precision = sum(precision_array)
    sum_f1 = sum(f1_array)
    sum_recall = sum(recall_array)
    
    return -weighted_precision
    
MAX_EVALS = 1000

# def _constant_metric(dy_true, dy_pred):
#     """An eval metric that always returns the same value"""
#     metric_name = 'constant_metric'
#     value = 0.708
#     is_higher_better = False
#     return metric_name, value, is_higher_better

if __name__ == "__main__":
    Correct = Total = TP = TN = FP = FN =0
    
    printOverall = True
    if(printOverall): 
        overall = open(f'overall_sm_{MODEL_NAME}.txt','w+')
        overall_M = pd.Series()
        overall_pred = pd.Series()
    
    for area in (0,1,2,3,4,5,6,7):
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

        target_M = train_data['label_M']
        train_feature = train_data.drop(['label_M','label_long','label_lati','Day'],axis=1)

        #震级类别化
        for i, ss in enumerate(target_M):
            if(ss<3.5):
                target_M.iloc[i] = 0
            elif(ss<4.0):
                target_M.iloc[i] = 1
            elif(ss<4.5):
                target_M.iloc[i] = 2
            elif(ss<5.0):
                target_M.iloc[i] = 3
            else:
                target_M.iloc[i] = 4
        #震级赋权
        train_data['weight'] = None
        train_data['label_M'] = target_M
        train_data['weight'][train_data['label_M']==0] = 1
        train_data['weight'][train_data['label_M']==1] = 1
        train_data['weight'][train_data['label_M']==2] = 1
        train_data['weight'][train_data['label_M']==3] = 1
        train_data['weight'][train_data['label_M']==4] = 1
        weight_T = train_data['weight'].values
    #验证数据
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
        
        trn_data = lgb.Dataset(train_feature, label=target_M,weight=weight_T)
        val_data = lgb.Dataset(valid_feature, label=valid_M, weight=weight_V)
        
        print("\nSearching parameters......")
        
        params = {
            'num_iterations': 5000,
            'num_leaves': 256,
            'learning_rate': 0.05,
            "boosting": "gbdt",
            'objective': 'multiclass', #转为分类问题
            'num_class': 5,
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
            "num_leaves":hp.choice("num_leaves",range(32,1024)),
            "feature_fraction":hp.uniform("feature_fraction",0.5,1.0),
            "bagging_fraction":hp.uniform("bagging_fraction",0.5,1.0),
            "bagging_freq":hp.choice("bagging_freq",range(2,10)),
            "min_child_samples":hp.choice("min_child_samples", range(10,50)),
        }
        
        #optimization
        best = fmin(fn=optimize, 
                    space=spaces, 
                    algo=tpe.suggest, 
                    max_evals = MAX_EVALS,
                    trials=Trials())
        best_params = space_eval(spaces, best)
        
        #train
        print("\nTraining......")
        params.update(best_params)
        clf = lgb.train(params, trn_data, valid_sets=[trn_data,val_data],
                        verbose_eval=500,early_stopping_rounds=10) #feval=_constant_metric,...

        #评估
        oof_lgb = np.matrix(clf.predict(valid_feature, num_iteration=clf.best_iteration))
        pred = pd.Series(clf.predict(valid_feature, num_iteration=clf.best_iteration).argmax(1))
        cpf = open(str(area)+f'_sm_{MODEL_NAME}.txt','w+')
        ccc = oof_lgb.argmax(axis=1)
        for i in range(len(ccc)):
            print(f"{i}, pre:{ccc[i]}, origin:{valid_M[i]}", file=cpf)
        
        correct = (oof_lgb.argmax(axis=1)==(np.matrix(valid_M).T)).sum()
        total = len(oof_lgb)
        tp = (np.array(oof_lgb.argmax(axis=1)!=0) * np.array((np.matrix(valid_M).T)!=0)).sum()
        tn = (np.array(oof_lgb.argmax(axis=1)==0) * np.array((np.matrix(valid_M).T)==0)).sum()
        fp = (np.array(oof_lgb.argmax(axis=1)!=0) * np.array((np.matrix(valid_M).T)==0)).sum()
        fn = (np.array(oof_lgb.argmax(axis=1)==0) * np.array((np.matrix(valid_M).T)!=0)).sum()
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
        print(f"\naccuracy: {accuracy}, {correct}/{total}")
        print(f"sensitivity(recall): {sensitivity}, {tp}/({tp}+{fn})")
        print(f"specificity(presicion): {specificity}, {tp}/({tp}+{fp})")
        print(f"f1: {f1}")
        print(classification_report(valid_M, pred, labels=range(0,5)))
        print(f"\naccuracy: {accuracy}, {correct}/{total}", file=cpf)
        print(f"sensitivity(recall): {sensitivity}, {tp}/({tp}+{fn})", file=cpf)
        print(f"specificity(precision): {specificity}, {tp}/({tp}+{fp})", file=cpf)
        print(f"f1: {f1}", file=cpf)
        print(classification_report(valid_M, pred, labels=range(0,5)), file=cpf)
        cpf.close()
        
        if(printOverall):
            print(classification_report(valid_M, pred, labels=range(0,5)), file=overall)
            overall_M = overall_M.append(valid_M)
            overall_pred = overall_pred.append(pred)
        
        #保存区域模型
        clf.save_model('./model/'+str(area)+f'_mag_{MODEL_NAME}_model.txt')

    Accuracy = Correct / Total
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1 = 2*TP / (2*TP+FP+FN)
    print(f"\n\nAccuracy: {Accuracy}, {Correct}/{Total}")
    print(f"recall: {Recall}, {TP}/({TP}+{FN})")
    print(f"precision: {Precision}, {TP}/({TP}+{FP})")
    print(f"F1: {F1}")
    if(printOverall):
        print(f"\n\nAccuracy: {Accuracy}, {Correct}/{Total}", file=overall)
        print(f"recall: {Recall}, {TP}/({TP}+{FN})", file=overall)
        print(f"precision: {Precision}, {TP}/({TP}+{FP})", file=overall)
        print(f"F1: {F1}", file=overall)
        print(classification_report(overall_M, overall_pred, labels=range(0,5)))
        print(classification_report(overall_M, overall_pred, labels=range(0,5)), file=overall)
        overall.close()
