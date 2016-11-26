from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
import nltk
import operator

lines = []
fin = open("../data/alice_in_wonderland.txt", "rb")
for line in fin:
    line = line.strip().decode("ascii", "ignore").encode("utf-8")
    if len(line) == 0:
        continue
    lines.append(line)
fin.close()

stop_words = set(nltk.corpus.stopwords.words("english"))
sentences = []
for sent in nltk.sent_tokenize(" ".join(lines)):
    sentence = []
    for word in nltk.word_tokenize(sent):
        if word in stop_words:
            continue
        sentence.append(word.lower())
    sentences.append(sentence)
    
model = Word2Vec(sentences, size=300, sg=1, window=3, min_count=5, workers=4)
model.init_sims(replace=True)

most_similar = model.most_similar("alice")
most_similar = [x[0] for x in 
    sorted(most_similar, key=operator.itemgetter(1), reverse=True)]
print("most_similar(alice): {:s}".format(", ".join(most_similar)))    
print("sim(alice, queen)", model.similarity("alice", "queen"))
print("sim(alice, puppy)", model.similarity("alice", "puppy"))
print("model['alice'].shape", model["alice"].shape)

model.save("word2vec_gensim.bin")
