import random
import numpy as np


def get_raw_data(file_dir = r"tang.npz"):
    data = np.load(file_dir)
    data, word2idx, idx2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
    #把每个整理成<START>,<EOP>的格式
    out_data = []
    for val in data:
        val = list(val)
        pos = val.index(word2idx["<START>"])
        #out_data.append(val[pos: len(val)])
        out_data.append(val)
    return out_data, word2idx, idx2word

class Poem_DataSet():
    def __init__(self):
        self.data, self.word2idx, self.idx2word = get_raw_data()
        self.len = len(self.data)

    def fetch_data(self, batch_size, mode="train"):
        random.shuffle(self.data)
        cnt = 0
        out_X = []
        out_Y = []
        for k in range(len(self.data)):
            out_X.append(self.data[k][0 : -1])
            out_Y.append(self.data[k][1 : len(self.data[k])])
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                yield out_X, out_Y
                out_X = []
                out_Y = []


if __name__ == "__main__":
    '''DS = Poem_DataSet()
    for idx, (X,Y) in enumerate(DS.fetch_data(batch_size=1)):
        print(X)
        print(Y)
        break'''
    data, word2idx, idx2word = get_raw_data()
    sum = 0
    calc_word = {}
    for poem in data:
        for word in poem:
            if word == 8292:
                continue
            sum += 1
            if word not in calc_word.keys():
                calc_word[word] = 0
            calc_word[word] += 1

    sorted_word = sorted(calc_word.items(), key = lambda x: x[1], reverse=True)
    cnt = 0
    for a in range(len(sorted_word)):
        cnt += sorted_word[a][1]
        if cnt >= 0.99 * sum:
            print(a)
            break
        print(idx2word[sorted_word[a][0]], sorted_word[a][1])
