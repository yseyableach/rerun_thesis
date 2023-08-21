# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 18:01:50 2020

@author: ccu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

train  = pickle.load(open("final_after_bert_train.pkl", 'rb'))
test   = pickle.load(open("final_after_bert_test.pkl", 'rb'))

train['kind_of_sentence'] = train['kind_of_sentence'].replace(["第一句", "內文", "最後一句"], [1,2,3]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
#train['categorical'] = train['categorical'].replace(["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"], [1,2,3,4,5]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
train_X = train.drop(["categorical","corpus","preprocessing_corpus"],axis=1)                                     #這邊這樣對不對 跟老師再確認一下
train_y = train['categorical']       


test['kind_of_sentence'] = test['kind_of_sentence'].replace(["第一句", "內文", "最後一句"], [1,2,3]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
#test['categorical'] = test['categorical'].replace(["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"], [1,2,3,4,5]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
test_X = test.drop(["categorical","corpus","preprocessing_corpus"],axis=1)                                     #這邊這樣對不對 跟老師再確認一下
test_y = test['categorical']      


#%% 這個部分應該先把 test分裝
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif #也可以用p-value 或 啥軌的有三個就對了
from sklearn.multiclass import OneVsOneClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn_crfsuite import CRF, scorers, metrics
crf = CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
# other scikit-learn modules
estimator = lgb.LGBMClassifier(num_leaves=16)

logic = LogisticRegression(n_jobs=1, C=1e5)
param_grid = {
    'learning_rate': [0.01, 0.1, 1,0.2,0.3,0.4,0.5,0.35],
    'n_estimators': [20, 40,30,70]
}
gbm = GridSearchCV(estimator, param_grid, cv=4)

xgb = xgb.XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=1000, 
                        num_classes=5,learning_rate=0.01 ,)

clf = OneVsOneClassifier(logic).fit(train_X[['locate','kind_of_sentence']], train_y)

#於54句改變logic 變稠 xgb gbm等分類方法
crf.fit(train_X, train_y)

y_pred_LG = clf.predict(test_X)
y_pred_LG = logic.predict(test_X) #超過0.5判斷1 小於則0
test_LGy=logic.predict_proba(test_X)#他判斷得機率 上課講到我們可以讓他更嚴格像是我得像0.99巷周杰倫才為周杰倫
print('accuracy %s' % accuracy_score(y_pred_LG, test_y))

from sklearn.metrics import f1_score
f1_score(y_pred_LG, test_y, average='macro')
from sklearn.metrics import classification_report

target_names = ["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"]

print(classification_report(test_y, y_pred_LG, target_names=target_names))




#%%
import xgboost as xgb
xgb = xgb.XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=1000, 
                        num_classes=5,learning_rate=0.01 ,)
xgb.fit(train_x2, train_y)  
y_pred = clf.predict_proba(train_X)

print(confusion_matrix(train_y, y_pred))


