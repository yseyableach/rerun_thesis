# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 11:50:28 2021

@author: ccu
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import nltk ,sklearn 
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
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


from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial.distance import cdist
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
## for bert language model
#import transformers
import pickle
import pandas as pd
train  = pickle.load(open("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/train_Pubmed_20k_didnot_vectorize_just_data_.pkl", 'rb'))
test   = pickle.load(open("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/test_Pubmed_20k_didnot_vectorize_just_data_.pkl", 'rb'))

train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

bio_train = pd.DataFrame(pickle.load(open("C:/Users/ccu/Downloads/bio_bert_sum_v1.1_vector_train_pubmed_20k_rct.pkl", 'rb')))
bio_test =  pd.DataFrame(pickle.load(open("C:/Users/ccu/Downloads/bio_bert_sum_v1.1_vector_test_pubmed_20k_rct.pkl", 'rb')))


bio_train.columns  =[("bio_"+str(i)) for i in range(768)]
bio_test.columns=[("bio_"+str(i)) for i in range(768)]


freq_word  = pd.read_json("C:/Users/ccu/Desktop/遊程傑/pubmed-rct-master/PubMed_20k_RCT_numbers_replaced_with_at_sign/Freq_Vocab_more_than_1000.json",encoding='big5') 

#%%
from gensim.models import Word2Vec
w2v = Word2Vec.load("C:/Users/ccu/Desktop/何沂蓁/DIM_200_pubmed200k/pubmed200k_corpus_200.model")

#%%
corpus = train["preprocessing_corpus"][0].split()
def sentecn_sum_by_googlenews(df):
    space=[]
    for corpus in df:  
        corpuss=corpus.split()
        d=np.zeros(200)
        for vocab in corpuss:
            try:
                a = w2v.wv[vocab]
                d=d+a     
                
                if(vocab in list(freq_word.freq_word.values)):
                    a = w2v.wv[vocab]
                    d=d+a
                    print("幹你娘")                       
                    
            except KeyError:
                print("跳過"+vocab+"這個字") 
        space.append(d)
    return pd.DataFrame(space)

train_w2v = sentecn_sum_by_googlenews(train["preprocessing_corpus"])
test_w2v = sentecn_sum_by_googlenews(test["preprocessing_corpus"])

train_w2v.columns  =[("fast_"+str(i)) for i in range(200)]
test_w2v.columns=[("fast_"+str(i)) for i in range(200)]



test['kind_of_sentence'] = test['kind_of_sentence'].replace(["第一句", "內文", "最後一句"], [1,2,3]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
train['kind_of_sentence'] = train['kind_of_sentence'].replace(["第一句", "內文", "最後一句"], [1,2,3]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3

train_w2v_location = pd.concat([train_w2v,train[['locate', 'kind_of_sentence', 'B','O', 'M', 'R', 'C']]],axis=1)
test_w2v_location = pd.concat([test_w2v,test[['locate', 'kind_of_sentence', 'B','O', 'M', 'R', 'C']]],axis=1)
#%%

train_X,train_y =train_w2v_location , train['categorical']
test_X,test_y = test_w2v_location, test['categorical']

big_train = pd.concat([ train['categorical'] ,train_w2v ],axis=1)
big_test = pd.concat([ test['categorical'] ,test_w2v ],axis=1)



def take_cosine_dis_from_corpus(train_df , test_df , dim):
    print("沙小?")
    B_corpus = train_df[train_df.categorical == "BACKGROUND"].iloc[:,1:]
    O_corpus = train_df[train_df.categorical == "OBJECTIVE"].iloc[:,1:]
    M_corpus = train_df[train_df.categorical == "METHODS"].iloc[:,1:]
    R_corpus = train_df[train_df.categorical == "RESULTS"].iloc[:,1:]
    C_corpus = train_df[train_df.categorical == "CONCLUSIONS"].iloc[:,1:]
    print(B_corpus)
    space = []
    #test_df=test_df.drop("categorical" , axis=1)
    
    print(test_df)
    for i in range(len(test_df)):        
        B_dis = cosine_similarity(B_corpus.values, test_df.loc[i].values.reshape(1,dim)).sum()/len(B_corpus)
        O_dis = cosine_similarity(O_corpus.values, test_df.loc[i].values.reshape(1,dim)).sum()/len(O_corpus)
        M_dis = cosine_similarity(M_corpus.values, test_df.loc[i].values.reshape(1,dim)).sum()/len(M_corpus)
        R_dis = cosine_similarity(R_corpus.values, test_df.loc[i].values.reshape(1,dim)).sum()/len(R_corpus)
        C_dis = cosine_similarity(C_corpus.values, test_df.loc[i].values.reshape(1,dim)).sum()/len(C_corpus)
        all_dis =[B_dis,O_dis,M_dis,R_dis , C_dis]
        space.append(all_dis)
    return pd.DataFrame(space,columns=["B_dis","O_dis","M_dis","R_dis","C_dis"])

train_cosine_dis_Matrix = take_cosine_dis_from_corpus(big_train,big_train.iloc[:,1:201] ,200)
test_cosine_dis_Matrix = take_cosine_dis_from_corpus(big_test,big_test.iloc[:,1:201] ,200)
 


#train_X = pd.concat([train_w2v,train_cosine_dis_Matrix,train[["locate","kind_of_sentence", 'B','O', 'M', 'R', 'C']]],axis=1)
#test_X = pd.concat([test_w2v,test_cosine_dis_Matrix,test[["locate","kind_of_sentence",'B','O', 'M', 'R', 'C']]],axis=1)


train_X = pd.concat([train_w2v,bio_train,train[["locate","kind_of_sentence", 'B','O', 'M', 'R', 'C']]],axis=1)
test_X = pd.concat([test_w2v,bio_test,test[["locate","kind_of_sentence",'B','O', 'M', 'R', 'C']]],axis=1)

#以下是方法選擇
#train_X =train_w2v
#test_X =test_w2v


#%% GBM
estimator = lgb.LGBMClassifier(num_leaves=16)
param_grid = {
    'learning_rate': [0.01, 0.1, 1,0.2,0.3,0.4,0.5,0.35],
    'n_estimators': [20, 40,30,70]
}
gbm = GridSearchCV(estimator, param_grid, cv=4)
#%% Logic
logic = LogisticRegression(n_jobs=1, C=1e5)
#%% XGB
import xgboost as xgb
xgb = xgb.XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=100, 
                        num_classes=5,learning_rate=0.01 ,)
#%% RF
rf = RandomForestClassifier(max_depth=5,n_estimators=80, random_state=0)
#%%

def evolution(method , M_name):
    method.fit(train_X, train_y)
    scores = cross_val_score(method, train_X, train_y, cv=10 , scoring='f1_weighted' )
    print("十哲驗證的 f1-macro :" ,str(scores) )
    print("ten cross validation the best scores is : " , max(list(scores)))
    y_pred = method.predict(test_X)
    target_names = ["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"]
    print("目前的演算法為 : ",M_name)
    print(classification_report(test_y, y_pred, target_names=target_names))
    print("##############################################################################")
    print("##############################################################################")
    
# other scikit-learn modules
evolution(gbm , "LightBGM")
evolution(logic , "Logic Regression")
evolution(xgb , "XGBoost")
evolution(rf , "Random Forest")
#%% crf



def take_categorical_from_jmir3inone(df):
    final_y = []
    aent_attr = []
    for i in range(len(df)): #2ㄏ
        if(i==0):
            aent_attr.append(df.categorical[i])  #第一句不能丟阿 美女
        else:        
            
            if(df.locate[i]!=1):    #判斷是不適第一句
                aent_attr.append(df.categorical[i])
            else:
                final_y.append(aent_attr) #不適的話你切一下可不可以?
                aent_attr = []
    return final_y



#p_train_y = take_categorical_from_jmir3inone(train)
#p_test_y = take_categorical_from_jmir3inone(test.reset_index())

y_final_train = take_categorical_from_jmir3inone(train)
y_final_test = take_categorical_from_jmir3inone(test)

#%%


def take_attr_from_jmir_to_dict(df,df2) :

    final_attr = []
    aent_attr = []
    for i in range(len(df)): #2ㄏ
        if(i==0):
            aent_attr.append(dict(df.iloc[i]))  #第一句不能丟阿 美女
        else:        
            if(df2.locate[i]!=1):    #判斷是不適第一句
                aent_attr.append(dict(df.iloc[i]))
            else:
                final_attr.append(aent_attr) #不適的話你切一下可不可以?
                aent_attr = []
    return final_attr

# train_X.columns = ["fast_"+str(i) for i in range(200)]
# test_X.columns = ["fast_"+str(i) for i in range(200)]

x_final_train = take_attr_from_jmir_to_dict(train_X , train)
x_final_test = take_attr_from_jmir_to_dict(test_X , test)

#%%
#train_X , trian_y = attr_final_jmir[0:5000], y_final_jmir[0:]
#test_X , test_y = attr_final_jmir[5000:], y_final_jmir[5000:]

crf_train_X , crf_trian_y = x_final_train , y_final_train
crf_test_X , crf_test_y = x_final_test , y_final_test
#%%   

def crf() :
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import make_scorer
    import scipy.stats
    crf = sklearn_crfsuite.CRF(
        algorithm='ap',
        max_iterations=100,
        all_possible_transitions=True
    )
    params_space = {
        'max_iterations':[10,50,100,200],
        'epsilon' : [1e-5,1e-6,1e-4,1e-7]
        
    }
    labels =  ["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"]    
    f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)
    
    rs = RandomizedSearchCV(crf, params_space,
                            cv=3,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=50,
                            scoring=f1_scorer)
    rs.fit(crf_train_X, crf_trian_y)
    print(rs.best_params_)
    crf = rs.best_estimator_
    crf.fit(crf_train_X, crf_trian_y)    
    y_pred = crf.predict(crf_test_X)
    print('best CV score:', rs.best_score_)
    print("CRF 預測結果") 
    print(metrics.flat_classification_report(
        crf_test_y, y_pred,  digits=3
    ))

crf()

print("sent+fasttext"+結果拉幹你娘機掰)
#%%

#%%
'''
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

max_features = 20000  # Only consider the top 20k words
maxlen = 300  # Only consider the first 200 words of each movie review
# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="float32")
# Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 128)(inputs)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()
#%%
#%%
train['categorical'] =  train['categorical'].replace(['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS','CONCLUSIONS'], [1, 2, 3, 4,5])

#%%
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
     train_w2v_location,train['categorical'], test_size=0.33, random_state=42)

X_train = keras.preprocessing.sequence.pad_sequences(X_train.values, maxlen=maxlen)
X_val = keras.preprocessing.sequence.pad_sequences(X_val.values, maxlen=maxlen)


model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train.values, batch_size=32, epochs=2, validation_data=(X_val, y_val.values))
'''