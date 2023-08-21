# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:27:30 2021

@author: ccu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os


os.system("cd C:/Users/ccu/Downloads")

bio_train  = pickle.load(open("bio_bert_v1.1_vector_train_pubmed_20k_rct.pkl", 'rb'))
bio_test   = pickle.load(open("bio_bert_v1.1_vector_test_pubmed_20k_rct.pkl", 'rb'))

name_list = []
for i in range(768):
    name="bio_"+str(i)
    name_list.append(name)
bio_train.columns = name_list
bio_test.columns = name_list

#%%
train  = pickle.load(open("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/處理過後的train/final_after_bert_train.pkl", 'rb'))
test   = pickle.load(open("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/處理過後的train/final_after_bert_test.pkl", 'rb'))
train = train.drop(['corpus','preprocessing_corpus'],axis=1)
test = test.drop(['corpus','preprocessing_corpus'],axis=1)
test['kind_of_sentence'] = test['kind_of_sentence'].replace(["第一句", "內文", "最後一句"], [1,2,3]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
train['kind_of_sentence'] = train['kind_of_sentence'].replace(["第一句", "內文", "最後一句"], [1,2,3]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
#%%



final_train = train.iloc[:,0:8].merge(bio_train, left_index=True, right_index=True)
final_test = test.iloc[:,0:8].merge(bio_test, left_index=True, right_index=True)

#%%
X_train , y_train = final_train.iloc[:,8:] , final_train.categorical
X_test , y_test = final_test.iloc[:,8:] , final_test.categorical

#%%

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
from sklearn.ensemble import RandomForestClassifier
#from sklearn_crfsuite import CRF, scorers, metrics
'''
crf = CRF(
    algorithm='lbfgs',
    c1=0.01,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
'''
# other scikit-learn modules
estimator = lgb.LGBMClassifier(num_leaves=16)

logic = LogisticRegression(n_jobs=1, C=1e5)
param_grid = {
    'learning_rate': [0.01, 0.1, 1,0.2,0.3,0.4,0.5,0.35],
    'n_estimators': [20, 40,30,70]
}
gbm = GridSearchCV(estimator, param_grid, cv=4)

xgb = xgb.XGBClassifier(max_depth=20, objective='multi:softprob', n_estimators=1000, 
                        num_classes=5,learning_rate=0.01 ,)

rf = RandomForestClassifier(max_depth=200, random_state=0)
clf = OneVsOneClassifier(rf).fit(X_train, y_train)


xgb.fit(X_train, y_train)
logic.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbm.fit(X_train, y_train)


#於54句改變logic 變稠 xgb gbm等分類方法
#crf.fit(train_X, train_y)

y_pred_xgb = xgb.predict(X_test)

y_pred_lr = logic.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_gbm = gbm.predict(X_test)




#y_pred_LG = logic.predict(test_X) #超過0.5判斷1 小於則0
#y_pred_LG = gbm.predict(test_X) #超過0.5判斷1 小於則0
#y_pred_LG = xgb.predict(test_w2v)

#test_LGy=logic.predict_proba(test_w2v_location)#他判斷得機率 上課講到我們可以讓他更嚴格像是我得像0.99巷周杰倫才為周杰倫
#print('accuracy %s' % accuracy_score(y_pred_LG, y_test))


#%%
from sklearn.metrics import classification_report
target_names = ["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"]
print(classification_report(y_test, y_pred_lr, target_names=target_names))
