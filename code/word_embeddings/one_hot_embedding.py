# -*- coding: utf-8 -*-
from keras.preprocessing.text import *
from sklearn.preprocessing import OneHotEncoder

texts = ["I am Sam .",
        "Sam I am .",
        "I love green eggs and ham ."]

tokenizer = Tokenizer(lower=False)
tokens = tokenizer.fit_on_texts(texts)

print(tokenizer.word_counts)

ohe = OneHotEncoder(n_values=len(tokenizer.word_counts))
for text in texts:
    embedding = one_hot(text, len(tokenizer.word_counts),
                        filters="", lower=False)
    print(text, embedding)
    print(ohe.fit_transform([embedding]).todense())
