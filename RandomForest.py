#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:24:47 2021

@author: max
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold #交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.model_selection import train_test_split #将数据集分开成训练集和测试集
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from sklearn.metrics import r2_score
import sklearn.utils as uts  # 打乱数据
import warnings
warnings.filterwarnings('ignore')
import sys
import json
import traceback
import joblib

# traindata_path = '/home/FBH/Bearing_Data/feature1.0/Bearing1_1_fea.csv'
# testdata_path = '/home/FBH/Bearing_Data/feature1.0/Bearing1_7_fea.csv'

# data1 = pd.DataFrame(pd.read_csv(traindata_path))
# data7 = pd.DataFrame(pd.read_csv(testdata_path))

# X1 = data1.drop(['Label'], axis=1) 
# X7 = data7.drop(['Label'], axis=1)

# Y1 = data1.loc[:,['Label']] 
# Y7 = data7.loc[:,['Label']] 

# # 数据正态分布归一化
# from sklearn.preprocessing import StandardScaler
# X_scaler = StandardScaler()
# X1_scaler =  X_scaler.fit_transform(X1)
# X7_scaler =  X_scaler.fit_transform(X7)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

class Result:
    precision = 0
    recall = 0
    accuracy = 0
    rocX = []
    rocY = []
    featureImportances = []
params = {}
params['n_estimators'] = 180 #1000
params['max_depth'] = 13
params['max_features'] = 2
params['min_samples_split'] = 15
params['min_samples_leaf'] = 10
params['train'] = '/Users/max/Desktop/BearingData/feature_smote/traindata_N15_M07_F10_fea.csv'
params['test'] = '/Users/max/Desktop/BearingData/feature/traindata_N15_M01_F10_fea.csv'
argvs = sys.argv
try:
    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])

    #训练集
    train = np.array(pd.read_csv(params['train']))
    train_y = train[:, -1]
    train_x = train[:, :-1]
    # train = pd.DataFrame(pd.read_csv(params['train']))
    # train_y = train['label']
    # train_x = train.loc[:, :'ratio_cD1']
    train_x, train_y = uts.shuffle(train_x, train_y, random_state=12)  # 打乱样本

    #测试集
    test = np.array(pd.read_csv(params['test']))
    test_y = test[:, -1]
    test_x = test[:, :-1]
    # test = pd.DataFrame(pd.read_csv(params['test']))
    # test_y = test['label']
    # test_x = test.loc[:, :'ratio_cD1']
    
    # x = pd.concat((train_x,test_x),axis=0)
    # y = pd.concat((train_y,test_y),axis=0)
    
    train_x = preprocessing.MinMaxScaler().fit_transform(train_x)
    test_x = preprocessing.MinMaxScaler().fit_transform(test_x)
    # x = preprocessing.MinMaxScaler().fit_transform(x)
    # x, y = uts.shuffle(x, y, random_state=15)  # 打乱样本
    #划分训练和验证集
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.23, random_state=14)

    clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                             max_features=params['max_features'],
                             max_depth=params['max_depth'],
                             min_samples_split=params['min_samples_split'],
                             min_samples_leaf=params['min_samples_leaf'],
                             random_state=10,
                             oob_score=True).fit(train_x, train_y)

    predict = clf.predict(test_x)
    precision = precision_score(test_y, predict, average='macro')
    recall = recall_score(test_y, predict, average='macro')
    accuracy = accuracy_score(test_y, predict)
    # predict = clf.predict(X_test)
    # precision = precision_score(y_test, predict, average='macro')
    # recall = recall_score(y_test, predict, average='macro')
    # accuracy = accuracy_score(y_test, predict)
    
    res = {}
    res['precision'] = precision
    res['recall'] = recall
    res['accuracy'] = accuracy
    res['fMeasure'] = f1_score(test_y, predict, average='macro')
    # res['fMeasure'] = f1_score(y_test, predict, average='macro')
    #res['rocArea'] = roc_auc_score(test_y, predict, average='macro', multi_class='ovo')
    # res['featureImportances'] = clf.feature_importances_.tolist()
    print(json.dumps(res))
    joblib.dump(clf,'/Users/max/Desktop/BearingData/RandomForest_model2.model')
    print('袋外验证分数：')
    print(clf.oob_score_)
    
    
except Exception as e:
    traceback.print_exc()
    print(e)
















