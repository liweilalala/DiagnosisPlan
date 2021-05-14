#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:43:52 2021

@author: max
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
import sys
import json
import traceback
import joblib

class Result:
    precision = 0
    recall = 0
    accuracy = 0
    rocX = []
    rocY = []
    featureImportances = []

params = {}
params['model'] = '/Users/max/Desktop/BearingData/RandomForest_model2.model'
params['test'] = '/Users/max/Desktop/BearingData/feature/test_fea2.csv'
params['opath'] = '/Users/max/Desktop/BearingData/test_upload.csv'

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

    model = joblib.load(params['model'])

    test_csv = pd.read_csv(params['test'])
    test_feature = test_csv.drop(['label'], axis=1)
    
    data = np.array(test_feature)
    number = data.shape[0]
    test_feature = preprocessing.MinMaxScaler().fit_transform(test_feature)
    predict = model.predict_proba(test_feature)
    predict = model.predict(test_feature)
    Predict = [i for i in range(number)]
    for i in range(number):
        # Predict = predict[i]
        Predict[i] = int(predict[i])
    
    
    # col_lab = ['predict']
    # result = pd.DataFrame(Predict, columns = col_lab)
    result = pd.DataFrame(Predict)
    result.to_csv(params['opath'], index=False)
    # print('Model_prediction is finished!')
    
    precision = precision_score(test_csv['label'], Predict, average='macro')
    recall = recall_score(test_csv['label'], Predict, average='macro')
    accuracy = accuracy_score(test_csv['label'], Predict)

    res = {}
    res['precision'] = precision
    res['recall'] = recall
    res['accuracy'] = accuracy
    res['fMeasure'] = f1_score(test_csv['label'], Predict, average='macro')
    res['rocArea'] = 0
    # res['featureImportances'] = model.feature_importances_.tolist()
    print(json.dumps(res))
except Exception as e:
    traceback.print_exc()
    print(e)














    