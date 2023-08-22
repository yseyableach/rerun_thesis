from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import AutoTokenizer, AutoModel
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
import torch,os
import pandas as pd 
import numpy as np

def train_and_save_doc2vec_model(corpus:list,corpus_name:str, vector_size=300, window=5, min_count=1, epochs=20, save_dir="embedding_models/doc2vec_models"):
    '''create doc2vec model'''
    os.makedirs(save_dir, exist_ok=True)
    tagged_data = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(corpus)]
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=4, epochs=epochs)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    model_filename = os.path.join(save_dir, f"{corpus_name}_dim_{vector_size}_doc2vec_model.model")
    model.save(model_filename)
    print(f"Model saved as {model_filename}")
    # return model

def train_and_save_fasttext_model(corpus:list, corpus_name:str, vector_size=300, window=5, min_count=1, epochs=20, save_dir="embedding_models/fasttext_models"):
    """create fast text model"""
    os.makedirs(save_dir, exist_ok=True)
    model = FastText(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, workers=4, sg=1, epochs=epochs)
    model_filename = os.path.join(save_dir, f"{corpus_name}_dim_{vector_size}_fasttext_model.model")
    model.save(model_filename)
    print(f"Model saved as {model_filename}")
    # return model

def get_biobert(method: str, corpus: list):
    # 載入BioBERT的tokenizer和model
    '''create biober embedding'''
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
    sentence_vectors = []
    for sentence in corpus:
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            if method == "sum":
                sentence_vector = torch.sum(outputs.last_hidden_state, dim=1).squeeze().numpy()
            elif method == "mean":
                sentence_vector = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
            else:
                raise ValueError("Invalid method. Supported methods: 'sum', 'mean'")
            sentence_vectors.append(sentence_vector)
    return np.array(sentence_vectors)



def get_wiki_news_embedding(corpus: list, model_file: str):
    # 載入預訓練的詞向量模型
    print("這個wiki 要load 很久 很扯== 大概5分鐘")
    wiki_word_vectors = KeyedVectors.load_word2vec_format(model_file)
    sentence_vectors = []
    for sentence in corpus:
        words = sentence.split()
        valid_words = [word for word in words if word in wiki_word_vectors]
        
        if len(valid_words) == 0:
            sentence_vector = np.zeros(wiki_word_vectors.vector_size)
        else:
            word_vectors_list = [wiki_word_vectors[word] for word in valid_words]
            sentence_vector = np.mean(word_vectors_list, axis=0)
        sentence_vectors.append(sentence_vector)
    return np.array(sentence_vectors)

#%% Train doc2vec & Train Fasttext
def generate_model(generate_corpus:str):
    corpus = None
    corpus_name= None
    if generate_corpus=='jmir':
        corpus_name="jmir"
        jmir_df = pd.read_csv("jmir/new_final_jmir_vertical.csv")
        corpus = jmir_df['pre_sent'].values.tolist()
        
    if generate_corpus=='pubmed':
        corpus_name ='pubmed'
        pubmed_df = pd.read_csv('pubmed20k/train_Pubmed_didnot_vectorize_just_data_.csv',encoding='big5')
        
        corpus = pubmed_df['preprocessing_corpus'].values.tolist()
    

    train_and_save_doc2vec_model(corpus,corpus_name,vector_size=50)
    print("Model trained and saved.")
    vector_size = [50,100,200]
    for size in vector_size:
        train_and_save_fasttext_model(corpus, corpus_name,vector_size=size)
    
# main()