# -*- coding: utf-8 -*-
from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams
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

# compute skip-gram pairs from last sentence
windex = {x[1]:x[0] for x in tokenizer.word_index.items()}
X, y = skipgrams(embedding, len(tokenizer.word_counts))
print(len(X), len(y))
for i in range(10):
    print("([{:s} ({:d}), {:s} ({:d})], {:d})"
        .format(windex[X[i][0]], X[i][0], windex[X[i][1]], X[i][1], y[i]))