import os
import re
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from underthesea import word_tokenize
from underthesea import text_normalize

# Define the path to the stopwords file
stopWordsPath = r'C:\Users\trand\Documents\study\university\23_24_ki1\streamlit_career\MODEL\vietnamese-stopwords.txt'

# Khởi tạo biến counter
counter = 0

def preprocessing(text):
    global counter  # Thêm dòng này để sử dụng biến counter

    text_pre = text.replace("\n", "")
    text_pre = re.sub(r'[^\w\s]', '', text_pre)
    text_pre = text_pre.lower()
    text_pre = re.sub(r'[^\w\s]', '', text_pre)
    text_pre = re.sub("\d+", " ", text_pre)
    text_pre = re.sub(r"[!@#$[]()]'-", "", text_pre)
    
    text_pre = word_tokenize(text_pre, format="text")
    text_pre = text_normalize(text_pre)
    
    f = open(stopWordsPath, "r", encoding="utf-8")
    
    List_StopWords = f.read().replace(' ', '_')
    List_StopWords = List_StopWords.split("\n")
    text_pre = " ".join(text for text in text_pre.split() if text not in List_StopWords)
    
    stop = stopwords.words('english')
    text_pre = " ".join(text_pre for text_pre in text_pre.split() if text_pre not in stop)
    
    counter += 1  # Tăng biến counter sau mỗi lần gọi hàm
    if counter % 100 == 0:
        print(f"Processed {counter} rows")
    
    return text_pre.split()
