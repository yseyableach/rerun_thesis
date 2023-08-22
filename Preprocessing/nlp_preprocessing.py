import re,nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
# 下載 WordNet
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')

def process_sentence(sentence:str):
    # 斷句
    sentences = sent_tokenize(sentence)
    # 初始化英文停用詞集合
    stop_words = set(stopwords.words('english'))
    # 初始化 WordNet 詞形還原器
    lemmatizer = WordNetLemmatizer()
    processed_sentences = []
    for sent in sentences:
        # 移除標點符號並轉為小寫
        sent = re.sub(r'[^a-zA-Z0-9\s]', '', sent.lower())
        # 斷詞
        words = sent.split()
        # 移除停用詞並進行詞形還原
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        if filtered_words:
            processed_sentences.append(" ".join(filtered_words))
    return processed_sentences

def get_BOMRC(category:str):
    '''generatte bomrc columns'''
    if category =='BACKGROUND':
        return [1,0,0,0,0]
    elif category =='OBJECTIVE':
        return [1,1,0,0,0]
    elif category =='METHODS':
        return [1,1,1,0,0]    
    elif category =='RESULTS':
        return [1,1,1,1,0]    
    elif category =='CONCLUSIONS':
        return [1,1,1,1,1]