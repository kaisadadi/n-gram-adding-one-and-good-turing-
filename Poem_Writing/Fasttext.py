#import fasttext
from gensim.models import word2vec, Word2Vec

s = word2vec.LineSentence("data.txt")
model = word2vec.Word2Vec(s, min_count=1, size=128, window=5)
model.save("Word_Vec.pkl")
#model = Word2Vec.load("Word_Vec")

import pickle

data = pickle.load(open("Word_Vec.pkl", "rb"))

#for i in model.most_similar("舟"): #计算余弦距离最接近“滋润”的10个词
#    print(i[0],i[1])

print(model)
print(model["花"])
