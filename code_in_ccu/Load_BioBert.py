# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:59:25 2020

@author: ccu
"""


import torch

from transformers import BertTokenizer, BertModel

import argparse
import logging
import torch
import os
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
logging.basicConfig(level=logging.INFO)


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)
    
os.chdir("C:/Users/ccu/Downloads")  
#convert_tf_checkpoint_to_pytorch("biobert_v1.1_pubmed/model.ckpt-1000000", "biobert_v1.1_pubmed/bert_config.json", "biobert_v1.1_pubmed/pytorch_model.bin")
model_version = 'biobert_v1.1_pubmed'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)


#C:\Users\ccu\Downloads
#step1 : 到biobert的github上下載所要使用的biobert版本
#step2 : 切換到該目錄底下執行上面這個函數convert_tf_checkpoint_to_pytorch
#step3 : 把這個檔案改名子 原因呢? 不明 。!mv biobert_v1.1_pubmed/bert_config.json biobert_v1.1_pubmed/config.json
#step4 : 他是以資料夾的形式讀取檔案 故只需要給他資料夾的位置。
#reference : https://www.kaggle.com/doggydev/biobert-embeddings-demo#BioBERT-Embeddings-Analysis

#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os

train  = pickle.load(open("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/train_Pubmed_20k_didnot_vectorize_just_data_.pkl", 'rb'))
test   = pickle.load(open("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/test_Pubmed_20k_didnot_vectorize_just_data_.pkl", 'rb'))
valid   = pickle.load(open("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/valid_Pubmed_20k_didnot_vectorize_just_data_.pkl", 'rb'))


train_sen = train['preprocessing_corpus']
test_sen = test['preprocessing_corpus']
valid_sen = valid['preprocessing_corpus']




#new_train  = pickle.load(open("bio_bert_v1.1_vector_train_pubmed_20k_rct.pkl", 'rb'))
#new_test   = pickle.load(open("bio_bert_v1.1_vector_test_pubmed_20k_rct.pkl", 'rb'))
#train_01 = train_sen.iloc[0:int(len(train_sen)*0.1)]

#%%



#%%
#(跑好久)超過3小時
def get_embedding_from_biobert_sum_6_layer(df):  #sentence 用 series的型態放入唷
    space=[]

    for i in range(len(df)):
        input_ids = torch.tensor(tokenizer.encode(df[i])).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the f
        #print(last_hidden_states)
        #print(last_hidden_states.shape)
        space.append(torch.mean(torch.sum(last_hidden_states, dim=0),dim=0).tolist())
        if((i%10000)==0):
            print(i,"已完成",str(i/len(df)))
#    data_corpus = pd.DataFrame(space)
    return space


#return pd.DataFrame(space)

new_train = get_embedding_from_biobert_sum_6_layer(train_sen)
new_test =  get_embedding_from_biobert_sum_6_layer(test_sen)
new_valid =  get_embedding_from_biobert_sum_6_layer(valid_sen)



pd.DataFrame(new_test)

import pickle 
pickle.dump(new_train, open("bio_bert_sum_v1.1_vector_train_pubmed_20k_rct.pkl", 'wb'))#這個是save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pickle.dump(new_test, open("bio_bert_sum_v1.1_vector_test_pubmed_20k_rct.pkl", 'wb'))#這個是save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pickle.dump(new_valid, open("bio_bert_sum_v1.1_vector_valid_pubmed_20k_rct.pkl", 'wb'))#這個是save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

'''
#%%
import pickle


bio_train = pickle.load(open("bio_bert_v1.1_vector_train_pubmed_20k_rct.pkl", 'rb'))
bio_test =  pickle.load(open("bio_bert_v1.1_vector_test_pubmed_20k_rct.pkl", 'rb'))


train['kind_of_sentence'] = train['kind_of_sentence'].replace(["第一句", "內文", "最後一句"], [1,2,3]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
test['kind_of_sentence'] = test['kind_of_sentence'].replace(["第一句", "內文", "最後一句"], [1,2,3]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3




train_X = pd.concat([train[['locate','kind_of_sentence','B','O','M','R','C']] ,new_train ],axis=1)
train['categorical'] = train['categorical'].replace(["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"], [1,2,3,4,5]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
train_y = train['categorical']       


test_X = pd.concat([test[['locate','kind_of_sentence','B','O','M','R','C']] ,new_test ],axis=1)
test['categorical'] = test['categorical'].replace(["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"], [1,2,3,4,5]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
test_y = test['categorical']        






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

from xgboost import XGBClassifier

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

param_dist = {'objective':'binary:logistic', 'n_estimators':2}

xgb = XGBClassifier(n_estimators=100)

clf = OneVsOneClassifier(xgb).fit(train_X.values, train_y.values)

#於54句改變logic 變稠 xgb gbm等分類方法
#crf.fit(train_X, train_y)

y_pred_LG = clf.predict(test_X)
#y_pred_LG = logic.predict(test_X) #超過0.5判斷1 小於則0
test_LGy=logic.predict_proba(test_X)#他判斷得機率 上課講到我們可以讓他更嚴格像是我得像0.99巷周杰倫才為周杰倫
print('accuracy %s' % accuracy_score(y_pred_LG, test_y))


from sklearn.metrics import f1_score
f1_score(y_pred_LG, test_y, average='macro')
from sklearn.metrics import classification_report

target_names = ["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"]
print(classification_report(test_y, y_pred_LG, target_names=target_names))
'''