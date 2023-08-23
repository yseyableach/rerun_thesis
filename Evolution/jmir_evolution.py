import pandas as pd 
import sys,pickle
sys.path.append('D:/GitHub_的東西/rerun_thesis/Dataset')
from embedding_model_genaratiom import get_biobert,get_wiki_news_embedding
from gensim.models.doc2vec import Doc2Vec
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from openpyxl import Workbook
from datetime import datetime
from sklearn_crfsuite import CRF
'''
setting what kind of embedding you want to choose 
doc2vec
wiki-news
biobert
'''

embedding_type = 'wiki-news'
dateset_name = 'jmir'
now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#%%

def merge_lists(lists):
    merged_list = []
    for sublist in lists:
        merged_list.extend(sublist)
    return merged_list

def crf_report_training(new_train_df_x,new_train_all_y,new_test_df_x,new_test_all_y):
    '''training crf Note : y_train can't use float need string'''
    # sorted_labels = ['BACKGROUND','OBJECTIVE','METHODS','RESULTS','CONCLUSIONS']
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    crf.fit(new_train_df_x, new_train_all_y)
    labels = list(crf.classes_)
    y_pred = crf.predict(new_test_df_x)
    y_pred_ = merge_lists(y_pred) 
    y_test_ = merge_lists(new_test_all_y)
    report = classification_report(
        y_pred_, y_test_, target_names=labels, output_dict=True,labels=['BACKGROUND','OBJECTIVE','METHODS','RESULTS','CONCLUSIONS']
    )        
    return report



def transfrom_trainng_or_test_pandas_to_crf_type(get_df):
    '''put train_Data or test_Data inro CRF Type'''
    new_transform_df = []
    new_transform_all_y = []
    grouped = get_df.groupby('journalId')
    for journal_id, group_data in grouped:
        space = []
        new_y = []
        for i in range(len(group_data)):
            result = group_data.iloc[i].to_dict()
            del result['journalId']
            del result['category']
            
            space.append(result)
            new_y.append(group_data.iloc[i]['category'])
            
        new_transform_df.append(space)
        new_transform_all_y.append(new_y)
    return {
        'X' : new_transform_df,
        'y' : new_transform_all_y
        }
    
def get_embedding(embedding_type:str,
                  dateset_name:str,
                  corpus:list,
                  vector_size = 50,
                  method='sum'):
    '''wikinews & biobert is pretrain modle , so we can just load from jay develop function'''

    if embedding_type == 'doc2vec':
        model_filename = f"../Dataset/embedding_models/doc2vec_models/{dateset_name}_dim_{vector_size}_{embedding_type}_model.model"
        saved_model = Doc2Vec.load(model_filename)
        result = [saved_model.infer_vector(sent.split()) for sent in corpus[0:3]]
        return pd.DataFrame(result)
        
    if embedding_type == 'fasttext':
        model_filename = f"../Dataset/embedding_models/doc2vec_models/{dateset_name}_dim_{vector_size}_{embedding_type}_model.model"
        saved_model = Doc2Vec.load(model_filename)
        result = [saved_model.infer_vector(sent.split()) for sent in corpus[0:3]]
        return pd.DataFrame(result)
    
    elif embedding_type == 'wiki-news':
        model_file = "../Dataset/embedding_models/wiki-news-300d-1M.vec"
        sentence_vectors = get_wiki_news_embedding(corpus, model_file)
        return pd.DataFrame(sentence_vectors)
    
    elif embedding_type =='biobert':
        sentence_vectors = get_biobert(method, corpus)
        return pd.DataFrame(sentence_vectors)

#%%

def perform_classification(X_train, X_test, y_train, y_test):
    
    '''compare four algorith auc'''
    # target_names=['BACKGROUND','OBJECTIVE','METHODS','RESULTS','CONCLUSIONS']
    algorithms = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    trained_models = {}  # 存放已訓練的模型
    for algorithm_name, algorithm in algorithms.items():
        algorithm.fit(X_train, y_train)
        y_pred = algorithm.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True,labels=[0,1,2,3,4])
        results[algorithm_name] = report
        # 將模型儲存起來

        model_filename = f"ClassiFicationModel/{now}_{algorithm_name}_{dateset_name}_{embedding_type}.pkl"
        trained_models[algorithm_name] = model_filename
        
        with open(model_filename, 'wb') as model_file:
            pickle.dump(algorithm, model_file)
            
    return results

def save_results_to_excel(results, excel_filename):
  '''save the result to xlsx'''
  workbook = Workbook()
  with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
      writer.book = workbook
      for algorithm_name, result in results.items():
          df = pd.DataFrame(result).reset_index()
          df.to_excel(writer, sheet_name=algorithm_name,index=False)
      workbook.save(excel_filename)


#%% main

# model_filename = "../Dataset/embedding_models/doc2vec_models/jmir_dim_50_doc2vec_model.model"
# saved_model = Doc2Vec.load(model_filename)
# model_filename = "embedding_models/fasttext_models/jmir_dim_50_fasttext_model.model"
# fasttext_model = FastText.load("path_to_your_fasttext_model")


jmir_df = pd.read_csv("../Dataset/jmir/new_final_jmir_vertical.csv")

corpus = jmir_df['pre_sent'].values.tolist()
corpus_embedding = get_embedding(embedding_type,dateset_name,corpus)
corpus_embedding.columns = [f"embedding_{i}" for i in corpus_embedding.columns]

get_othersfeature = jmir_df[['journalId','category']]
get_final_data = pd.concat([get_othersfeature,corpus_embedding],axis=1)
get_final_data.to_csv(f'embeddingDataset/{embedding_type}_{dateset_name}_jay.csv',index=False)


#%%
class_to_index = {
    'BACKGROUND': 0,
    'OBJECTIVE': 1,
    'METHODS': 2,
    'RESULTS': 3,
    'CONCLUSIONS': 4
}

index_to_class = {
    0:'BACKGROUND',
    1:'OBJECTIVE',
    2:'METHODS',
    3:'RESULTS',
    4:'CONCLUSIONS',
}

get_final_data = pd.read_csv(f'embeddingDataset/{embedding_type}_{dateset_name}_jay.csv')
get_final_data['category'] = get_final_data['category'].apply(lambda x : class_to_index[x])
train_data = get_final_data[get_final_data['journalId']<4500]
test_data =  get_final_data[get_final_data['journalId']>=4500]
X_train, y_train = train_data.drop(['journalId','category'],axis=1) ,train_data['category']
X_test, y_test = test_data.drop(['journalId','category'],axis=1) ,test_data['category']
final_evolution = perform_classification(X_train[0:100], X_test[0:100], y_train[0:100], y_test[0:100])

#%%train的部分
train_data['category'] = train_data['category'].apply(lambda x : index_to_class[x])
test_data['category'] = test_data['category'].apply(lambda x : index_to_class[x])

crf_train = transfrom_trainng_or_test_pandas_to_crf_type(train_data)
crf_test = transfrom_trainng_or_test_pandas_to_crf_type(test_data)
crf_report = crf_report_training(crf_train['X'],
                                 crf_train['y'],
                                 crf_test['X'],
                                 crf_test['y'],
                                 )
final_evolution['CRF'] =  crf_report
save_results_to_excel(final_evolution,f'experiment/{now}_{embedding_type}_{dateset_name}_evolution.xlsx')