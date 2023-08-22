# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 21:57:14 2021

@author: ccu
"""
import matplotlib.pyplot as plt
from itertools import chain
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
#%%
#讀入資料
import pickle
nltk.corpus.conll2002.fileids()
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
train  = pickle.load(open("final_after_bert_train.pkl", 'rb'))
test   = pickle.load(open("final_after_bert_test.pkl", 'rb'))

#crf因為是list丟進去部會自己判斷dimention的0~767為str 所以要自己幫他取名子
for i in range(0,768): #bert 所產生的維度
    name = "bert_dim_"+str(i)
    train = train.rename(columns={i: name})    
    test = test.rename(columns={i: name})        
    print(i)
    
#%%train的部分
train_t=train.iloc[:,1:]
new_train_df=[]
new_train_all_y=[]
new_y=[]
space=[]
for i in range(len(train_t)) :
    if(train_t['kind_of_sentence'][i]=="最後一句"):
        space.append(train_t.loc[i].to_dict())
        new_y.append(train['categorical'][i])
        new_train_df.append(space)
        new_train_all_y.append(new_y)
        space=[]
        new_y=[]
        #將這句以後的新增到new_df
    else:
        space.append(train_t.loc[i].to_dict())
        new_y.append(train['categorical'][i])
   
#%%test的部分
test_t=test.iloc[:,1:]
new_test_df=[]
new_test_all_y=[]
new_y=[]
space=[]
for i in range(len(test_t)) :
    if(test_t['kind_of_sentence'][i]=="最後一句"):
        space.append(test_t.loc[i].to_dict())
        new_y.append(train['categorical'][i])
        new_test_df.append(space)
        new_test_all_y.append(new_y)
        space=[]
        new_y=[]
        #將這句以後的新增到new_df
    else:
        space.append(test_t.loc[i].to_dict())
        new_y.append(train['categorical'][i])

#%%        
%%time

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(new_train_df, new_train_all_y)

#%%
y_pred = crf.predict(new_test_df)
metrics.flat_f1_score(new_test_all_y, y_pred,
                      average='weighted')

print(metrics.flat_classification_report(
    y_pred, new_test_all_y, digits=3
))

