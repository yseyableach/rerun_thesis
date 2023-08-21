
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os


bio_train  = pd.DataFrame(pickle.load(open("bio_bert_sum_v1.1_vector_train_pubmed_20k_rct.pkl", 'rb')))
bio_test   = pd.DataFrame(pickle.load(open("bio_bert_sum_v1.1_vector_test_pubmed_20k_rct.pkl", 'rb')))

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


final_train = pd.concat([train.iloc[:,0:8],bio_train],axis=1)
final_test = pd.concat([test.iloc[:,0:8],bio_train],axis=1)

#%%
'''
def sentecn_mean_by_googlenews(df):
    space=[]
    for corpus in df:  
        corpuss=corpus.split()
        d=np.zeros(300)
        time = 0 
        for vocab in corpuss:
            try:
                a = w2v.wv[vocab]
                d=d+a
                time = time + 1 
            except KeyError:
                print("跳過"+vocab+"這個字") 
        space.append(d/time)
    return pd.DataFrame(space)
jmir_fast = sentecn_mean_by_googlenews(jmir_data["preprocessing_corpus"])

new_name= []
for i in range(300):
    name="fast_"+str(i)
    new_name.append(name)
    
    
jmir_fast.columns = new_name
final_jmir = pd.concat([jmir_fast,jmir_data.iloc[:,1:9]],axis=1)
'''
#final_jmir = jmir_fast
#%%train_test_split
#split_len = 62519
#train , test =final_jmir.iloc[0:split_len] , final_jmir.iloc[split_len:]

#train_X , train_y , test_X , test_y = train.iloc[:,0:307] , train.categorical , test.iloc[:,0:307],test.categorical
#%%


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

y_final_train = take_categorical_from_jmir3inone(final_train)

y_final_test = take_categorical_from_jmir3inone(final_test)


#%%


def take_attr_from_jmir_to_dict(df):

    final_attr = []
    aent_attr = []
    for i in range(len(df)): #2ㄏ
        if(i==0):
            aent_attr.append(dict(df.iloc[i]))  #第一句不能丟阿 美女
        else:        
            if(df.locate[i]!=1):    #判斷是不適第一句
                aent_attr.append(dict(df.iloc[i]))
            else:
                final_attr.append(aent_attr) #不適的話你切一下可不可以?
                aent_attr = []
    return final_attr


x_final_train = take_attr_from_jmir_to_dict(final_train.drop("categorical",axis=1))
x_final_test = take_attr_from_jmir_to_dict(final_test.drop("categorical",axis=1))
#%%
#train_X , trian_y = attr_final_jmir[0:5000], y_final_jmir[0:]
#test_X , test_y = attr_final_jmir[5000:], y_final_jmir[5000:]

train_X , trian_y = x_final_train , y_final_train
test_X , test_y = x_final_test , y_final_test
#%%      
import sklearn_crfsuite
from sklearn_crfsuite import metrics
crf = sklearn_crfsuite.CRF(
    algorithm='arow',
    max_iterations=100,
    all_possible_transitions=True,
    gamma = 20,


)
crf.fit(train_X, trian_y)    
#%%  
y_pred = crf.predict(test_X)
#%%
print(metrics.flat_classification_report(
    test_y, y_pred,  digits=3
))

#%%
for i in x_final_test[0]:
    print(i)
#%%
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection  import cross_val_score
from sklearn.model_selection  import RandomizedSearchCV
# define fixed parameters and parameters to search
labels = ['BACKGROUND','OBJECTIVE','METHODS','RESULTS','CONCLUSIONS']
crf = sklearn_crfsuite.CRF(
    algorithm='arow',
    max_iterations=100,
    all_possible_transitions=True
)
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

# use the same metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)

# search
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)
rs.fit(train_X, trian_y)
    
crf = rs.best_estimator_

