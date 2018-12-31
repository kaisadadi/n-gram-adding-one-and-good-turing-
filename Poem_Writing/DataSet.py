import random
import numpy as np


def get_raw_data(file_dir = r"tang.npz"):
    data = np.load(file_dir)
    data, word2idx, idx2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
    #把每个整理成<START>,<EOP>的格式
    out_data = []
    for val in data:
        val = list(val)
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

class Seq_Dataset():
    def __init__(self, self_embedding = None):
        self.rawdata, self.word2idx, self.idx2word = get_raw_data()
        self.len = len(self.rawdata)
        start_sign = self.word2idx["<START>"]
        break_sign = [self.word2idx["，"], self.word2idx["。"], self.word2idx["<EOP>"]]
        self.data = []
        self.embedding = self_embedding
        #f = open("data.txt", "w", encoding="utf8")
        for poem in self.rawdata:
            temp_data = []
            pos = poem.index(start_sign)
            t1 = pos
            for move in range(pos+1, len(poem)):
                if poem[move] in break_sign and move - t1 >= 3:
                    temp_data.append(poem[t1+1 : move])
                    t1 = move
            for seq_num in range(len(temp_data) - 1):
                if len(temp_data[seq_num]) != 5 or len(temp_data[seq_num + 1]) != 5:
                    break
                #out_data = []
                #for a in temp_data[seq_num]:
                #    out_data.append(self.idx2word[a])
                #    out_data.append(" ")
                #f.writelines(out_data)
                #f.write("\n")
                self.data.append([temp_data[seq_num], temp_data[seq_num + 1]])
            '''out_data = []
            try:
                for a in temp_data[-1]:
                    out_data.append(self.idx2word[a])
                    out_data.append(" ")
                f.writelines(out_data)
                f.write("\n")
            except:
                print(temp_data)'''

        #f.close()

    def fetch_data(self, batch_size, self_embedding=False):
        random.shuffle(self.data)
        cnt = 0
        out_X = []
        out_Y = []
        for k in range(len(self.data)):
            out_X.append(self.data[k][0])
            out_Y.append(self.data[k][1])
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                if self_embedding == True:
                    yield out_X, out_Y
                else:
                    yield self.embeded(out_X), self.embeded(out_Y)
                out_X = []
                out_Y = []

    def embeded(self, x):
        for batch in range(len(x)):
            for idx in range(len(x[batch])):
                x[batch][idx] = self.embedding[self.idx2word[x[batch][idx]]]
        return x

if __name__ == "__main__":
    '''
    DS = Poem_DataSet()
    for idx, (X,Y) in enumerate(DS.fetch_data(batch_size=1)):
        print(X)
        print(Y)
        break
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
    '''
    gg = Seq_Dataset()
    for idx, val in enumerate(gg.fetch_data(batch_size=1)):
        print(val[0])
        print(val[1])
        break