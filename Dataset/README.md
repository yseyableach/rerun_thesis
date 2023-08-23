## prepare_jmir.py

This file can use fianl_jmir3inone.csv to generate new_final_vertical.csv 

```
cd Dataset
python prepare.jmir.py
```

## embedding_model_generation.py

This file can genreate Fasttext * Doc2vec model on the path below
- Dataset/embedding_models/doc2vec_models
- Dataset/embedding_models/fasttext_models

You only need to choose what kind of corpus you want to generate on embedding_model_generation.py

```
generate_model('jmir')
generate_model('pubmed')
```