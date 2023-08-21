# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:39:01 2020
@author: ccu
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertModel
train  = pickle.load(open("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/處理過後的train/final_after_bert_train.pkl", 'rb'))
test   = pickle.load(open("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/處理過後的train/final_after_bert_test.pkl", 'rb'))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')                 #選擇bert的斷詞 Load pre-trained model tokenizer (vocabulary)

###################################################################################################################
##
## 這邊的transform_for_doc2vec只是為了要將list裡面的單詞段詞，由於我過去寫的很棒 雇用空白分開即是斷詞
##
## 至於為啥要取FOR_DOC2VEC是一位我要符合GENSIM的格式 反正就降
#####################################################################################################################
def transform_for_doc2vec(list_name):
    space=[]
    for i in list_name:
        space.append(i.split() )
    return space
####################################################################################################################
#段詞好的段落
B = transform_for_doc2vec(list(train[train['categorical']=='BACKGROUND']['preprocessing_corpus']))
O = transform_for_doc2vec(list(train[train['categorical']=='OBJECTIVE']['preprocessing_corpus']))
M = transform_for_doc2vec(list(train[train['categorical']=='METHODS']['preprocessing_corpus']))
R = transform_for_doc2vec(list(train[train['categorical']=='RESULTS']['preprocessing_corpus']))
C = transform_for_doc2vec(list(train[train['categorical']=='CONCLUSIONS']['preprocessing_corpus']))
TRAIN_SPLIT = transform_for_doc2vec(list(train['preprocessing_corpus']))
#%%
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
#5轉成DOC2VEC我覺得TRAIN的還算快 顧 下面的再轉乘PKL就好。
B_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(B)]
B_model = Doc2Vec(B_documents, vector_size=50, window=2, min_count=1, workers=4)
##########################################################################################
#屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼
##################################################################################
O_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(O)]
O_model = Doc2Vec(O_documents, vector_size=50, window=2, min_count=1, workers=4)
##########################################################################################
#屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼
##################################################################################
M_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(M)]
M_model = Doc2Vec(M_documents, vector_size=50, window=2, min_count=1, workers=4)
##########################################################################################
#屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼
##################################################################################
R_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(R)]
R_model = Doc2Vec(R_documents, vector_size=50, window=2, min_count=1, workers=4)
##########################################################################################
#屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼
##################################################################################
C_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(C)]
C_model = Doc2Vec(C_documents, vector_size=50, window=2, min_count=1, workers=4)
##########################################################################################
#屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼屁眼
##################################################################################
OVA_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(TRAIN_SPLIT)] 
OVA_model     = Doc2Vec(OVA_documents, vector_size=250, window=2, min_count=1, workers=4)
###################################################################################################################################
#%%
def concat_the_one_vs_one(train_list):
    space = []
    for text in train_list:
        bvector = list(B_model.infer_vector(text))
        ovector = list(O_model.infer_vector(text))
        mvector = list(M_model.infer_vector(text))
        rvector = list(R_model.infer_vector(text))
        cvector = list(C_model.infer_vector(text))
        all_vec = bvector+ovector+mvector+rvector+cvector
        space.append(all_vec)
    return pd.DataFrame(space)

def concat_the_one_vs_ALL(train_list):
    space = []
    for text in train_list:
        all_vec = list(OVA_model.infer_vector(text)) 
        space.append(all_vec)
    return pd.DataFrame(space)

final_OVO_DOC2VEC = concat_the_one_vs_one(TRAIN_SPLIT)
final_OVO_DOC2VEC = pd.concat([final_OVO_DOC2VEC,train[['B','O','M','R','C','locate','categorical']]],axis=1 )
final_OVA_DOC2VEC = concat_the_one_vs_ALL(TRAIN_SPLIT)
final_OVA_DOC2VEC.insert(250,"categorical",train['categorical'].values)
#%%這邊就輪流測試ㄅ 記得改元dataset
from sklearn.model_selection import train_test_split
train_X,train_y =final_OVO_DOC2VEC.iloc[:,0:250][0:120000],final_OVO_DOC2VEC['categorical'][0:120000]
test_X,test_y =final_OVO_DOC2VEC.iloc[:,0:250][120000:],final_OVO_DOC2VEC['categorical'][120000:]
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
logic = LogisticRegression(n_jobs=1, C=1e5)

clf = OneVsOneClassifier(gbm).fit(train_X, train_y) #請在 OneVsOneClassifier(xxxx) 於 xxxx中換方法 如 gbm ,xgb ,clf 等
#於54句改變logic 變稠 xgb gbm等分類方法
y_pred_LG = clf.predict(test_X)
#y_pred_LG = logic.predict(test_X) #超過0.5判斷1 小於則0
#test_LGy=logic.predict_proba(test_X)#他判斷得機率 上課講到我們可以讓他更嚴格像是我得像0.99巷周杰倫才為周杰倫
print('accuracy %s' % accuracy_score(y_pred_LG, test_y))
from sklearn.metrics import f1_score
f1_score(y_pred_LG, test_y, average='macro')
from sklearn.metrics import classification_report
target_names = ["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"]
print(classification_report(test_y, y_pred_LG, target_names=target_names))
#%%








'''
這邊是SAVE MODEL的部分
fname = get_tmpfile("my_doc2vec_model")
model.save(fname)
model = Doc2Vec.load(fname)  # you can continue training with the loaded model!
'''


'''
#If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
'''

