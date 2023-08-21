# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:24:51 2020

@author: ccu
"""
#extract location + preprocessing + vecorrize


#%%

#%%sent_locat_extract 將Pubmed 20K RCT 的資料切用LIST一筆一筆將他裝起來
import pandas as pd
P_20k_RCT = pd.read_table('train.txt')
P_20k_RCT = (P_20k_RCT.reset_index()).fillna("next_abstract")
print("記得更換title 如果不知道我在說啥 請執行17行後 將論文流水編號取代")
rename_dic = {"index": "categorical" , "###24293578" :"corpus"} #切換文本的時候這邊季的改一下 看改成啥 
P_20k_RCT=P_20k_RCT.rename(rename_dic, axis=1)


final_P_20k_RCT = []
start=0
for i in range(len(P_20k_RCT)):
    if(P_20k_RCT['corpus'][i]=="next_abstract"):
        final_P_20k_RCT.append(P_20k_RCT[start:(i)])
        start = i+1
#這邊就完成了把它裝起來


#接著將句子的特徵逐一放進去       

the_final=[]   
for corpus in final_P_20k_RCT:
    locate=[]
    kind_of_sentence=[]
    label=['BACKGROUND','OBJECTIVE','METHODS','RESULTS','CONCLUSIONS']
    #用來存放判定該篇論文前面有沒有被判斷過
    B = []
    O = []
    M = []
    R = []
    C = []
    #判定過去的文章是否已經有被判定為該類別
    BB = False
    OO = False
    MM = False
    RR = False
    CC = False
    #這部分是用來判斷句子位置以及是否斷句
    corpus=corpus.reset_index()

    for i in range(len(corpus)):   
        print(i)
        locate.append(i+1)
        if(i==0):
            kind_of_sentence.append("第一句")
        elif(corpus['categorical'][i] in label and (i!=(len(corpus)-1))):
            kind_of_sentence.append("內文")
        else:
            print("安?")
            kind_of_sentence.append("最後一句")
            
            
    #BACKGROUND        
        if(("BACKGROUND" in corpus['categorical'][i] or BB)):
            if(BB==False):
                B.append(0)
                BB = True
            else:
                B.append(1)
        else:
            B.append(0)
    #OOJECTIVE     
        if(("OBJECTIVE" in corpus['categorical'][i] or OO )):
            if(OO==False):
                O.append(0)
                OO = True
            else:
                O.append(1)
        else:
            O.append(0)
    
    #METHODS     
        if(("METHODS" in corpus['categorical'][i] or MM )):
            if(MM==False):
                M.append(0)
                MM = True
            else:
                M.append(1)
        else:
            M.append(0)
            
    #RESULTS     
        if(("RESULTS" in corpus['categorical'][i] or RR )):
            if(RR==False):
                R.append(0)
                RR = True
            else:
                R.append(1)
        else:
            R.append(0)    
    #CONCLUSION
        if(("CONCLUSIONS" in corpus['categorical'][i] or CC )):
            if(CC==False):
                C.append(0)
                CC = True
            else:
                C.append(1)
        else:
            C.append(0)            
    print("fuck")      
    print(corpus)
    corpus.insert(2, 'locate', locate)
    corpus.insert(3, 'kind_of_sentence', kind_of_sentence)
    corpus.insert(4, 'B', B)
    corpus.insert(5, 'O', O)
    corpus.insert(6, 'M', M)
    corpus.insert(7, 'R', R)
    corpus.insert(8, 'C', C)
    corpus=corpus.drop("index" , axis = 1)
    the_final.append(corpus)
#%%pre_processing.py

import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
import nltk
import re
import pickle

from nltk.stem import WordNetLemmatizer
wtlem = WordNetLemmatizer()

ps = PorterStemmer()                                                           #我於2021/1/27 改成使用wordNet
stop_words = set(stopwords.words('english'))  
final_P_20k_R2CT  = the_final

                                                             
for i in range(len(final_P_20k_R2CT)):                                         #該層為我要選取第i篇學術文章摘要
    final_sent = [] 
    for j in final_P_20k_R2CT[i]['corpus']:                                    #該層為第i篇學術摘要中的第j個句子
        original_corpus = j.lower() #將文本小寫 
        original_corpus = re.sub(r'[^\w\s]','',original_corpus)                #去除符號 數字資料及本身已經刪除 
        corpus_tokenize = regexp_tokenize(original_corpus, pattern='\w+|\$[\d\.]+|\S+') #斷詞 本身已經斷句、故就不在斷句了，JMIR的資料集則需要在斷句 應該說 先斷句再丟到這也ok
        filtered_sentence = [w for w in corpus_tokenize if not w in stop_words]  #刪除停用詞
        sent =""                                                               #用來儲存鋸子的字串
        for c in range(len(filtered_sentence)):                                #Porter的steeming
            print(c)          
            
            filtered_sentence[c] = wtlem.lemmatize(filtered_sentence[c],'v')        #wordnet
            #filtered_sentence[c] = ps.stem(filtered_sentence[c])               #Porter的steeming
            sent=sent+str(filtered_sentence[c])                                #接著前面處理好的句子都整理好了 把它重新組合一下
            if(c != (len(filtered_sentence)-1)):                               #最後一句的時候就不要加上空格了
                sent=sent+" "                                                  #其他句的話還是要加上空格唷
            else:
                continue
        final_sent.append(sent)                                                #將處理好的鋸子塞進去
    #pre_sen_df = pd.DataFrame({"preprocessing_corpus":final_sent})
    final_P_20k_R2CT[i].insert(9,"preprocessing_corpus",final_sent)            #變成DataFrame塞進去
#這邊檢查一個 

AA=final_P_20k_R2CT[i].reset_index()
#這邊把資料拉直
data = pd.DataFrame()
print("資料拉直中 請稍後")
for i in range(len(final_P_20k_R2CT)):
    if(i%1000==0):
        print(i) #檢查城市有沒有再跑用的
    new_data=final_P_20k_R2CT[i]
    data=data.append(new_data) 
data = data.reset_index()
final_data = data.iloc[:,1:]
#%%
import pickle
pickle.dump(final_data, open("train_Pubmed_20k_didnot_vectorize_just_data_.pkl", 'wb'))#這個是save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#qweqwe  = pickle.load(open("the_final.pkl", 'rb'))
#final_data.to_csv("train_Pubmed_200k_didnot_vectorize_just_data_.csv",encoding='big5')

#%%bert_vectorize
import pickle
import pandas as pd 
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens') #biobert_v1.1_pubmed    distilbert-base-nli-stsb-mean-tokens
final_P_20k_data  = final_data
#Our sentences we like to encode
all_sentence = final_P_20k_data['preprocessing_corpus']
all_sss = all_sentence.tolist()
#Sentences are encoded by calling model.encode()
embeddings = model.encode(all_sss)#4:31開始的 大概跑一小時 
#embeddings to DataFrame
final_DF = pd.DataFrame(embeddings)
final_after_bert = pd.concat([final_P_20k_data,final_DF],axis=1)
#%%以上就處理完了唷 棒棒