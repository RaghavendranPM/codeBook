# -*- coding: utf-8 -*-
from gensim.models import word2vec
from nltk.corpus import stopwords
import nltk
import operator
import os
import sys
import logging

#lines = []
#fin = open("../data/alice_in_wonderland.txt", "rb")
#for line in fin:
#    line = line.strip().decode("ascii", "ignore").encode("utf-8")
#    if len(line) == 0:
#        continue
#    lines.append(line)
#fin.close()
#
#stop_words = set(nltk.corpus.stopwords.words("english"))
#sentences = []
#for sent in nltk.sent_tokenize(" ".join(lines)):
#    sentence = []
#    for word in nltk.word_tokenize(sent):
#        if word in stop_words:
#            continue
#        sentence.append(word.lower())
#    sentences.append(sentence)
#    
#model = Word2Vec(sentences, size=300, sg=1, window=3, min_count=5, workers=4)
#model.init_sims(replace=True)
#
#most_similar = model.most_similar("alice")
#most_similar = [x[0] for x in 
#    sorted(most_similar, key=operator.itemgetter(1), reverse=True)]
#print("most_similar(alice): {:s}".format(", ".join(most_similar)))    
#print("sim(alice, queen)", model.similarity("alice", "queen"))
#print("sim(alice, puppy)", model.similarity("alice", "puppy"))
#print("model['alice'].shape", model["alice"].shape)
#
#model.save("word2vec_gensim.bin")


#Lahiri, Shibamouli. "Complexity of word collocation networks: A preliminary structural analysis." arXiv preprint arXiv:1310.5111 (2013).


#class GutenbergSentences(object):
#    def __init__(self, dirname):
#        self.dirname = dirname
#    
#    def __iter__(self):
#        for fname in os.listdir(self.dirname):
#            print("reading file: {:s}".format(fname))
#            fin = open(os.path.join(self.dirname, fname), "rb")
#            texts = []
#            for line in fin:
#                line = line.strip()
#                try:
#                    line = line.decode("utf-8")
#                except UnicodeDecodeError:
#                    continue
#                if len(line) < 60:  # empty lines or TOC lines
#                    continue
#                if line == line.upper():  # all caps, copyright
#                    continue
#                texts.append(line)
#            # make single text of filtered file contents
#            text = " ".join(texts)
#            for sent in nltk.sent_tokenize(text):
#                words = [x.lower() for x in nltk.word_tokenize(sent)]
#                yield words
#
#
#DATA_DIR = "../data/Gutenberg/txt"
#
##logging.basicConfig(format="%(asctime)s: %(levelname)s : %(message)s", 
##                    level=logging.INFO)
#logging.basicConfig(level=logging.INFO)
#sentences = GutenbergSentences(DATA_DIR)
#model = Word2Vec(sentences, size=300, sg=1, window=3, min_count=5, workers=4)
#
#print(model.vocab)
#print(len(model.vocab))


class Text8Sentences(object):
    def __init__(self, fname, maxlen):
        self.fname = fname
        self.maxlen = maxlen
        
    def __iter__(self):
        with open(os.path.join(DATA_DIR, "text8"), "rb") as ftext:
            text = ftext.read().split(" ")
            sentences = []
            words = []
            for word in text:
                if len(words) >= self.maxlen:
                    yield words
                    words = []
                words.append(word)
            yield words

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DATA_DIR = "../data/"
sentences = Text8Sentences(os.path.join(DATA_DIR, "text8"), 50)
model = word2vec.Word2Vec(sentences, size=300, min_count=30)

print("""model.most_similar("woman")""")
print(model.most_similar("woman"))
#[('child', 0.7057571411132812),
# ('girl', 0.702182412147522),
# ('man', 0.6846336126327515),
# ('herself', 0.6292711496353149),
# ('lady', 0.6229539513587952),
# ('person', 0.6190367937088013),
# ('lover', 0.6062309741973877),
# ('baby', 0.5993420481681824),
# ('mother', 0.5954475402832031),
# ('daughter', 0.5871444940567017)]
 
print("""model.most_similar(positive=["woman", "king"], negative=["man"], topn=10)""")
print(model.most_similar(positive=['woman', 'king'], 
                         negative=['man'], 
                         topn=10))
#[('queen', 0.6237582564353943),
# ('prince', 0.5638638734817505),
# ('elizabeth', 0.5557916164398193),
# ('princess', 0.5456407070159912),
# ('throne', 0.5439794063568115),
# ('daughter', 0.5364126563072205),
# ('empress', 0.5354889631271362),
# ('isabella', 0.5233952403068542),
# ('regent', 0.520746111869812),
# ('matilda', 0.5167444944381714)]                         
                         
print("""model.similarity("girl", "woman")""")
print(model.similarity("girl", "woman"))
print("""model.similarity("girl", "man")""")
print(model.similarity("girl", "man"))
print("""model.similarity("girl", "car")""")
print(model.similarity("girl", "car"))
print("""model.similarity("bus", "car")""")
print(model.similarity("bus", "car"))
#model.similarity("girl", "woman")
#0.702182479574
#model.similarity("girl", "man")
#0.574259909834
#model.similarity("girl", "car")
#0.289332921793
#model.similarity("bus", "car")
#0.483853497748
