# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:18:53 2021

@author: Chen
"""
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import pickle , os
os.chdir("C:/Users/Chen/Downloads/biobert_v1.1_pubmed")
bio_bert = pd.read_csv("jmirt_bio.csv",encoding='utf8')

import pandas as pd , numpy as np ,pickle
jmir_data  = pickle.load(open("C:/Users/Chen/Desktop/論文的資料/jmir 相關處理/jmir3inone_.pkl", 'rb'))
jmir_data['kind_of_sentence'] = jmir_data['kind_of_sentence'].replace(["第一句", "內文", "最後一句"], [1,2,3]) #LabelEncoder 句子位置 將 第一句、內文、最後一句 改成 1,2,3
#%%

from gensim.models import Word2Vec
w2v = Word2Vec.load("C:/Users/Chen/Desktop/論文的資料/DIM_100_pubmed200k/pubmed200k_corpus_100.model")

#%%
corpus = jmir_data["preprocessing_corpus"].loc[0:5]

dim=100
def sentecn_sum_by_googlenews(df):
    space=[]
    for corpus in df:  
        corpuss=corpus.split()
        d=np.zeros(dim)
        for vocab in corpuss:
            try:
                a = w2v.wv[vocab]
                d=d+a     
                
            except KeyError:
                print("跳過"+vocab+"這個字") 
        space.append(d)
    return pd.DataFrame(space)

def get_distril_bert_embedding(all_sentence):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens') #biobert_v1.1_pubmed    distilbert-base-nli-stsb-mean-tokens
    all_sss = all_sentence.tolist()
    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(all_sss)#4:31開始的 大概跑一小時 
    #embeddings to DataFrame
    final_DF = pd.DataFrame(embeddings)
    final_DF.columns = ["DISTRIL_"+str(i) for i in range(768)]
    return final_DF


fast_test_jmir = sentecn_sum_by_googlenews(jmir_data['preprocessing_corpus'])

DISTRIL_BERT_test_jmir = get_distril_bert_embedding(jmir_data['preprocessing_corpus'])


fast_test_jmir.columns = ["fast_"+str(i) for i in range(dim)]

bio_bert.columns = ["bio_"+str(i) for i in range(769)]
bio_bert = bio_bert.drop("bio_0",axis=1)



fast = fast_test_jmir

fast_test_jmir = pd.concat([DISTRIL_BERT_test_jmir,jmir_data.iloc[:,1:9]],axis=1)



#%%
final_data = pd.concat([jmir_data.iloc[:,1:9],fast,bio_bert.iloc[:,1:]],axis=1)
final_data = pd.concat([fast,jmir_data.iloc[:,1:9]],axis=1)


final_data =pd.concat([fast,jmir_data.categorical],axis=1)

#from sklearn.model_selection import train_test_split
#train_X, test_X, train_y, test_y = train_test_split( final_data,jmir_data.categorical, test_size=0.2, random_state=42)
#train_X, test_X, train_y, test_y = train_test_split( final_jmir.iloc[:,0:300],final_jmir.categorical, test_size=0.2, random_state=42)
#all_data = pd.concat([train_X,train_y],axis=1)

split =int(len(fast_test_jmir)*0.8)+4
train = final_data.loc[0:split]
test =  final_data.loc[split+1:].reset_index().iloc[:,1:]


#%%
# train_X , test_X =pd.concat([train.iloc[:,0:200],train_cosine_dis_Matrix],axis=1),pd.concat([test.iloc[:,0:200],test_cosine_dis_Matrix],axis=1)

# train_X  , test_X = train.drop("categorical",axis=1) ,  test.drop("categorical",axis=1)         

train_X  , test_X = train.drop("categorical",axis=1) ,  test.drop("categorical",axis=1)         


train_y , test_y = train.categorical,test.categorical

train_X  , test_X = bio_bert.loc[0:split] , bio_bert.loc[split+1:].reset_index(drop=True)



#%% 句子相似度
def take_cosine_dis_from_corpus(train_df , test_df):
    B_corpus = train_df[train_df.categorical == "BACKGROUND"].iloc[:,0:200]
    O_corpus = train_df[train_df.categorical == "OBJECTIVE"].iloc[:,0:200]
    M_corpus = train_df[train_df.categorical == "METHODS"].iloc[:,0:200]
    R_corpus = train_df[train_df.categorical == "RESULTS"].iloc[:,0:200]
    C_corpus = train_df[train_df.categorical == "CONCLUSIONS"].iloc[:,0:200]
    space = []
    for i in range(len(test_df)):        
        B_dis = cosine_similarity(B_corpus.values, test_df.loc[i].values.reshape(1,200)).sum()/len(B_corpus)
        O_dis = cosine_similarity(O_corpus.values, test_df.loc[i].values.reshape(1,200)).sum()/len(O_corpus)
        M_dis = cosine_similarity(M_corpus.values, test_df.loc[i].values.reshape(1,200)).sum()/len(M_corpus)
        R_dis = cosine_similarity(R_corpus.values, test_df.loc[i].values.reshape(1,200)).sum()/len(R_corpus)
        C_dis = cosine_similarity(C_corpus.values, test_df.loc[i].values.reshape(1,200)).sum()/len(C_corpus)
        all_dis =[B_dis,O_dis,M_dis,R_dis , C_dis]
        space.append(all_dis)        
    return pd.DataFrame(space,columns=["B_dis","O_dis","M_dis","R_dis","C_dis"])

cos_train = train[["fast_"+str(i) for i in range(200)]]
cos_test = test[["fast_"+str(i) for i in range(200)]]


def cosine_get_big(df):
    for i in range(len(df)):
        df.loc[i]= df.loc[i].apply(lambda x: 1 if x == df.loc[i].max() else 0)
    print(df)
    return df


train_cosine_dis_Matrix = cosine_get_big(take_cosine_dis_from_corpus(train,cos_train))
test_cosine_dis_Matrix = cosine_get_big(take_cosine_dis_from_corpus(test,cos_test))


train_cosine_ =take_cosine_dis_from_corpus(train,cos_train)
text_coeine_ =take_cosine_dis_from_corpus(test,cos_test)


train_cosine_all = pd.concat([train_cosine_dis_Matrix,train_cosine_,train.categorical],axis=1)
test_cosine_all = pd.concat([test_cosine_dis_Matrix,text_coeine_,train.categorical],axis=1)


train_cosine_all.to_csv("train_cosine_all.csv",index=False)
test_cosine_all.to_csv("test_cosine_all.csv",index=False)


'''
ans=[]
for col in train_cosine_dis_Matrix.columns:
    for i in range(len(train_cosine_dis_Matrix)) :
        if(train_cosine_dis_Matrix.loc[i][col]==train_cosine_dis_Matrix.loc[i].max()):
            print(col)
            ans.append(col)
       
ans=[]
for i in range(len(train_cosine_dis_Matrix)):
    for col in train_cosine_dis_Matrix.columns:
        
        if(train_cosine_dis_Matrix[col][i]==train_cosine_dis_Matrix.loc[i].max()):
            if(col=="B_dis"):
                ans.append("BACKGROUND")
            elif(col=="O_dis"):
                ans.append("OBJECTIVE")
            elif(col=="M_dis"):
                ans.append("METHODS")
            elif(col=="R_dis"):
                ans.append("RESULTS")
            elif(col=="C_dis"):
                ans.append("CONCLUSIONS")
            break
            

'''

#%% CRF的東西

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

y_final_jmir = take_categorical_from_jmir3inone(fast_test_jmir[['categorical','locate']])

#%%


def take_attr_from_jmir_to_dict(df ):

    final_attr = []
    aent_attr = []
    for i in range(len(df)): #2ㄏ
        if(i==0):
            aent_attr.append(dict(df.iloc[i]))  #第一句不能丟阿 美女
        else:        
            if(jmir_data.locate[i]!=1):    #判斷是不適第一句
                aent_attr.append(dict(df.iloc[i]))
            else:
                final_attr.append(aent_attr) #不適的話你切一下可不可以?
                aent_attr = []
    return final_attr


cos_ =pd.concat([train_cosine_dis_Matrix,test_cosine_dis_Matrix]).reset_index(drop=True)
train_test_cos = pd.concat([fast ,cos_ ],axis=1)


attr_final_jmir = take_attr_from_jmir_to_dict(pd.concat([train_X,test_X],axis=0).reset_index(drop=True))

attr_final_jmir = take_attr_from_jmir_to_dict(fast.reset_index(drop=True))


#attr_final_jmir = take_attr_from_jmir_to_dict(pd.concat([train_X,test_X],axis=0).reset_index().iloc[:,1:])

#%%
crf_train_X , crf_trian_y = attr_final_jmir[0:4080], y_final_jmir[0:4080]
crf_test_X , crf_test_y = attr_final_jmir[4080:], y_final_jmir[4080:]

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
from sklearn.neural_network import MLPClassifier
from sklearn_crfsuite import CRF, scorers, metrics
from sklearn.model_selection import cross_val_score


estimator = lgb.LGBMClassifier(num_leaves=16)
param_grid = {
    'learning_rate': [0.01, 0.1, 1,0.2,0.3,0.4,0.5,0.35],
    'n_estimators': [20, 40,30,70]
}
gbm = GridSearchCV(estimator, param_grid, cv=4)
#%% Logic
logic = LogisticRegression(n_jobs=1, C=1e5)
#%% XGB
xgb = xgb.XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=100, 
                        num_classes=5,learning_rate=0.01 ,)
#%% RF
rf = RandomForestClassifier(max_depth=5,n_estimators=80, random_state=0)
#%% mlp
mlp = MLPClassifier(random_state=1, max_iter=100)
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
evolution(mlp , "Multi layer Percentron")
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
    crf = rs.best_estimator_
    crf.fit(crf_train_X, crf_trian_y)    
    y_pred = crf.predict(crf_test_X)
    print('best CV score:', rs.best_score_)
    print("CRF 預測結果") 
    print(metrics.flat_classification_report(
        crf_test_y, y_pred,  digits=3
    ))

crf()

print("distril_bert 結果")

'''

# other scikit-learn modules
estimator = lgb.LGBMClassifier(num_leaves=16)

rf = RandomForestClassifier(max_depth=1000, random_state=0)

logic = LogisticRegression(n_jobs=1, C=1e5)

param_grid = {
    'learning_rate': [0.01, 0.1, 1,0.2,0.3,0.4,0.5,0.35],
    'n_estimators': [20, 40,30,70]
}
gbm = GridSearchCV(estimator, param_grid, cv=4)
param_dist = {'objective':'binary:logistic', 'n_estimators':2}

#xgb=xgb.XGBModel(**param_dist)

xgb = xgb.XGBClassifier(max_depth=20, objective='multi:softprob', n_estimators=1000, 
                        num_classes=5,learning_rate=0.01 ,)
xgb.fit(train_X.values, train_y.values)

mlp = MLPClassifier(random_state=1, max_iter=300).fit(train_X,train_y)
rf.fit(train_X.values, train_y.values)
gbm.fit(train_X,train_y)
logic.fit(train_X,train_y)



y_pred_xgb = xgb.predict(test_X.values)

y_pred_lr = logic.predict(test_X)

y_pred_rf = rf.predict(test_X)
y_pred_gbm = gbm.predict(test_X)

y_pred_mlp = mlp.predict(test_X)


#%%
#clf=logic
#clf.fit(train_X,train_y)



y_pred_lr = logic.predict_proba(test_X)

#y_pred_lg_proab = clf.predict_proba(test_X)
#y_pred_LG = logic.predict(test_X) #超過0.5判斷1 小於則0
#test_LGy=logic.predict_proba(test_X)#他判斷得機率 上課講到我們可以讓他更嚴格像是我得像0.99巷周杰倫才為周杰倫
print('accuracy %s' % accuracy_score(y_pred_LG, test_y))
from sklearn.metrics import f1_score
f1_score(y_pred_LG, test_y, average='macro')
#%%
y_pred_LG = rf.predict(test_X)

from sklearn.metrics import classification_report
target_names = ["BACKGROUND", "OBJECTIVE","METHODS","RESULTS","CONCLUSIONS"]
print(classification_report(test_y, y_pred_xgb, target_names=target_names))

#%%
from sklearn.metrics.pairwise import cosine_similarity
sen1_sen2_similarity =  cosine_similarity([list(train_X.loc[0])],[list(train_X.loc[1])])

#%%

from sklearn.metrics.pairwise import cosine_similarity


# Vectors
vec_a = [1, 2, 3, 4, 5]
vec_b = [1, 3, 5, 7, 9]

# Dot and norm
dot = sum(a*b for a, b in zip(vec_a, vec_b))
norm_a = sum(a*a for a in vec_a) ** 0.5
norm_b = sum(b*b for b in vec_b) ** 0.5

# Cosine similarity
cos_sim = dot / (norm_a*norm_b)

# Results
print('My version:', cos_sim)
print('Scikit-Learn:', cosine_similarity([vec_a,vec_b], [vec_b]))
#%%
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

x = np.random.rand(1,1000)
y = np.random.rand(1000,1000)
cosine_similarity(x, y).sum()

'''
target_names = ["BACKGROUND", "OBJECTIVE", "METHODS","RESULTS","CONCLUSIONS"]

print(classification_report(train_y, ans, target_names=target_names))

