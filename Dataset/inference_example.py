from gensim.models.doc2vec import Doc2Vec
from embedding_model_genaratiom import get_biobert

# 載入已儲存的模型
model_filename = "embedding_models/doc2vec_models/jmir_dim_50_doc2vec_model.model"
saved_model = Doc2Vec.load(model_filename)

# model_filename = "embedding_models/fasttext_models/jmir_dim_50_fasttext_model.model"
# fasttext_model = FastText.load("path_to_your_fasttext_model")

# 示例文本
example_text = "This is a test sentence."
# 將文本轉換為向量
vector = saved_model.infer_vector(example_text.split())
print("Text:", example_text)
print("Vector:", vector)

#%%
corpus = [
    "This is the first sentence.",
    "This is another sentence.",
    "An example sentence."
]

method = "sum"
sentence_vectors = get_biobert(method, corpus)
print("Method:", method)
print("Sentence Vectors:")
print(sentence_vectors)