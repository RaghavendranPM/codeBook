# -*- coding: utf-8 -*-
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split
import collections
import nltk
import numpy as np

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

# Read training data and generate vocabulary
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
ftrain = open("data/umich-sentiment-train.txt", 'rb')
for line in ftrain:
    label, sentence = line.strip().split("\t")
    words = nltk.word_tokenize(sentence.decode("ascii", "ignore").lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1
ftrain.close()

## Get some information about our corpus
#print maxlen            # 42
#print len(word_freqs)   # 2313

# -1 is UNK, 0 is PAD
# We take MAX_FEATURES-1 featurs to accound for PAD
vocab = { "UNK" : -1, "PAD" : 0 }
reverse_vocab = { -1 : "UNK", 0 : "PAD" }
for idx, word in enumerate([w[0] for w in 
        word_freqs.most_common(MAX_FEATURES-1)]):
    vocab[word] = idx + 1
    reverse_vocab[idx + 1] = word

# convert sentences to sequences
X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0
ftrain = open("data/umich-sentiment-train.txt", 'rb')
for line in ftrain:
    label, sentence = line.strip().split("\t")
    words = nltk.word_tokenize(sentence.decode("ascii", "ignore").lower())
    seqs = []
    for word in words:
        if vocab.has_key(word):
            seqs.append(vocab[word])
        else:
            seqs.append(-1)
    X[i] = seqs
    y[i] = int(label)
    i += 1
ftrain.close()

# Split input into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, 
                                                random_state=42)
                                                
# Pad the sequences (left padded with zeros)
Xtrain = sequence.pad_sequences(Xtrain, maxlen=MAX_SENTENCE_LENGTH)
Xtest = sequence.pad_sequences(Xtest, maxlen=MAX_SENTENCE_LENGTH)

# Build model
model = Sequential()
model.add(Embedding(MAX_FEATURES, 128, input_length=MAX_SENTENCE_LENGTH,
                    dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=["accuracy"])

model.fit(Xtrain, ytrain, batch_size=32, nb_epoch=10, 
          validation_data=(Xtest, ytest))
score, acc = model.evaluate(Xtest, ytest, batch_size=32)

print("Test score: %.3f, accuracy: %.3f" % (score, acc))

for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([reverse_vocab[x] for x in xtest[0].tolist() if x != 0])
    print("%.3f\t%d\t%s" % (ypred, ylabel, sent))
