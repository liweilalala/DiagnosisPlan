#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 12:08:05 2021

@author: TeslaHuo
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

params = {}
params['data_path'] = r'F:\PythonProject\MobileNet\BearingProject\BearingData\feature\traindata_N15_M07_F10_fea2.csv'
params['save_path'] = r'F:\PythonProject\MobileNet\BearingProject\BearingData\smote_feature\traindata_N15_M07_F10_fea2.csv'
# X为特征，y为对应的标签
train = pd.DataFrame(pd.read_csv(params['data_path']))
y = train['label']
X = train.loc[:, :'ratio_cD1']

from collections import Counter
# 查看所生成的样本类别分布，0和1样本比例9比1，属于类别不平衡数据
print(Counter(y))
# Counter({0: 900, 1: 100})

# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
# 定义SMOTE模型，random_state相当于随机数种子的作用
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_sample(X, y)
y_smo=y_smo.reshape(-1,1)
#X_smo1=pd.DataFrame(X_smo)
#y_smo1=pd.DataFrame(y_smo)
#print(np.size(X_smo))
#print(np.size(y_smo))
smo_train = np.concatenate((X_smo,y_smo),axis=1)
# 查看经过SMOTE之后的数据分布
#print(Counter(y_smo))
# Counter({0: 900, 1: 900})

#col_lab = ['time_mean','time_std','time_max','time_min','time_rms','time_ptp','time_median','time_iqr','time_pr','time_sknew','time_kurtosis','time_var','time_amp','time_smr','time_wavefactor','time_peakfactor','time_pulse','time_margin',
#           'freq_mean','freq_std','freq_max','freq_min','freq_rms','freq_median','freq_iqr','freq_pr','freq_f2','freq_f3','freq_f4','freq_f5','freq_f6','freq_f7','freq_f8',
#           'ener_cA5','ener_cD1','ener_cD2','ener_cD3','ener_cD4','ener_cD5','ratio_cA5','ratio_cD1','ratio_cD2','ratio_cD3','ratio_cD4','ratio_cD5','label']
col_lab = ['time_mean','time_median','freq_mean','freq_std','freq_median','freq_f2','freq_f5','freq_f6','freq_f7','freq_f8','ener_cD1','ratio_cD1','label']
result = pd.DataFrame(smo_train, columns = col_lab)
result.to_csv(params['save_path'], sep=',', header=True, index=False)



