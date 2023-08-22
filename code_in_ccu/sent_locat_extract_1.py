
#%%將Pubmed 20K RCT 的資料切用LIST一筆一筆將他裝起來
import pandas as pd
P_20k_RCT = pd.read_table('test.txt')
P_20k_RCT = (P_20k_RCT.reset_index()).fillna("next_abstract")

rename_dic = {"index": "categorical" , "###24845963" :"corpus"} #切換文本的時候這邊季的改一下 看改成啥 
P_20k_RCT=P_20k_RCT.rename(rename_dic, axis=1)


final_P_20k_RCT = []
start=0
for i in range(len(P_20k_RCT)):
    if(P_20k_RCT['corpus'][i]=="next_abstract"):
        final_P_20k_RCT.append(P_20k_RCT[start:(i-1)])
        start = i+1
#這邊就完成了把它裝起來


#%%接著將句子的特徵逐一放進去       

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
'''   
import pickle
pickle.dump(the_final, open("the_final.pkl", 'wb'))#這個是save!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
qweqwe  = pickle.load(open("the_final.pkl", 'rb'))
'''