# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.layers.core import Activation, Dense, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
import nltk
import numpy as np
import seq2seq
from seq2seq.models import SimpleSeq2Seq

INPUT_FILE = "../data/alice_in_wonderland.txt"

# (1) extract text from file into list of words
words = []
fin = open(INPUT_FILE, 'rb')
for line in fin:
    line = line.strip()
    if len(line) == 0:
        continue
    for sentence in nltk.sent_tokenize(line):
        for word in nltk.word_tokenize(sentence):
            words.append(word.lower())
fin.close()

# (2) create lookup tables
char_vocab = set([c for c in " ".join(words)])
nb_chars = len(char_vocab)
char2index = dict((c, i) for i, c in enumerate(char_vocab))
index2char = dict((i, c) for i, c in enumerate(char_vocab))

# (3) create input and output texts

def reverse_words(words):
    reversed_words = []
    for w in words:
        reversed_words.append("".join(reversed([c for c in w])))
    return reversed_words

# ['very', 'tired', 'of', 'sitting']
# ['yrev', 'derit', 'fo', 'gnittis']
seqlen = 4
step = 1
input_texts = []
output_texts = []
for i in range(len(words) - seqlen):
    input_words = words[i:i + seqlen]
    input_texts.append(" ".join(input_words))
    output_texts.append(" ".join(reverse_words(input_words)))

maxlen = max([len(x) for x in input_texts])

# (4) vectorizing input and output texts
X = np.zeros((len(input_texts), maxlen, nb_chars), dtype=np.bool)
y = np.zeros((len(output_texts), maxlen, nb_chars), dtype=np.bool)
for i, input_text in enumerate(input_texts):
    input_text = input_text.ljust(maxlen)
    for j, ch in enumerate([c for c in input_text]):
        X[i, j, char2index[ch]] = 1
for i, output_text in enumerate(output_texts):
    output_text = output_text.ljust(maxlen)
    for j, ch in enumerate([c for c in output_text]):
        y[i, j, char2index[ch]] = 1

# (5) split data into training and validation
Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.1, 
                                              random_state=42)
    
## (5) build model
#model = Sequential()
#model.add(LSTM(512, input_shape=(maxlen, nb_chars), return_sequences=False))
#model.add(RepeatVector(maxlen))
#model.add(LSTM(512, return_sequences=True))
#model.add(TimeDistributed(Dense(nb_chars)))
#model.add(Activation("softmax"))
#
#model.compile(loss="categorical_crossentropy", optimizer="adam", 
#              metrics=["accuracy"])

# (5/alt) build model using seq2seq
model = SimpleSeq2Seq(input_shape=(maxlen, nb_chars),
                      hidden_dim=512, output_length=maxlen,
                      output_dim=nb_chars, unroll=True)
model.add(TimeDistributed(Dense(nb_chars)))
model.add(Activation("softmax"))                      
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])
              
# (6) test model
def decode_text(y):
    text_seq = []
    for i in range(y.shape[0]):
        idx = np.argmax(y[i])
        text_seq.append(index2char[idx])
    return "".join(text_seq).strip()

for iteration in range(50):
    print("=" * 50)
    print("Iteration-#: %d" % (iteration))
    model.fit(Xtrain, ytrain, batch_size=128, nb_epoch=1,
              validation_data=(Xval, yval))
    # select 10 samples from validation data
    for i in range(10):
        test_idx = np.random.randint(Xval.shape[0])
        Xtest = np.array([Xval[test_idx, :, :]])
        ytest = np.array([yval[test_idx, :, :]])
        ypred = model.predict([Xtest], verbose=0)
        xtest_text = decode_text(Xtest[0])
        ytest_text = decode_text(ytest[0])
        ypred_text = decode_text(ypred[0])
        print("input: [%s], expected: [%s], got: [%s]" % (
            xtest_text, ytest_text, ypred_text))
