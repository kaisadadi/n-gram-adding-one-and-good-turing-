from DataSet import Poem_DataSet
from torch.autograd import Variable
import torch.nn as nn
import torch
from DataSet import get_raw_data
import numpy as np
from LSTM_net import LSTM_net
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", "-g")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_path = "/home/wke18/Model"


def cross_entropy(X, Y):
    #print(X.shape)
    X = X.view(-1, 8300)
    Y = Y.view(-1, 1)
    #Y = torch.clamp(Y, max=5)
    X = nn.Softmax()(X)
    X = torch.clamp(X, 1e-6, 1 - 1e-6)
    pos = X.topk(10)
    #print(pos[0][0])
    #print(pos[0].shape)
    #print(pos[1][0])
    for a in range(X.shape[0]):
        sum = 0
        for val in X[a].topk(10)[0]:
            sum += val
        for pos in X[a].topk(10)[1]:
            X[a][pos] = sum
    Y_onehot = Variable(torch.zeros(Y.shape[0], 8300).scatter_(1, Y.cpu().data, 1)).cuda()
    #loss = torch.sum(-Y_onehot * torch.log(X) - (1 - Y_onehot) * torch.log(1 - X), dim=1).view(-1, 124)
    loss = torch.sum(-Y_onehot * torch.log(X), dim=1).view(-1, 124)
    return torch.mean(torch.sum(loss, dim=1))

def generate_poem(net, start_words, word2idx, idx2word, expected_length = 26):
    #为你写诗，为你静止
    #26代表经典的五言诗(4句)
    write_sequence = [word2idx["<START>"]]
    for word in start_words:
        write_sequence.append(word2idx[word])
    while len(write_sequence) < expected_length:
        #print(write_sequence)
        results, _ = net.forward(Variable(torch.from_numpy(np.array(write_sequence)).long().view(1, -1)).cuda())
        results = results.view(-1, 8300)
        result = results[-1].view(-1)
        prob = nn.Softmax()(result)
        write_sequence.append(int(torch.argmax(prob).cpu().data[0].numpy()))
    return write_sequence

def generate_seq(net, start_words, prefix, word2idx, idx2word, expected_length = 4):
    #从序列到序列的诗句
    if prefix != None:
        for idx, val in enumerate(prefix):
            prefix[idx] = word2idx[val]
        prefix = Variable(torch.from_numpy(np.array(prefix).long().view(1, -1))).cuda()
        _, hidden = net.forward(prefix)
    write_sequence = [[]]
    for word in start_words:
        write_sequence[-1].append(word2idx[word])
    while len(write_sequence) < expected_length:
        results, hidden = net.forward(Variable(torch.from_numpy(np.array(write_sequence[-1])).long().view(1, -1)).cuda(), hidden)
        write_sequence.append([])
        for idx in results:
            write_sequence[-1].append(idx)
    for idx1, sequence in enumerate(write_sequence):
        for idx2, val in enumerate(sequence):
            write_sequence[idx1][idx2] = idx2word(write_sequence[idx1][idx2])
    return write_sequence



def train_net(net, epoch, word2idx, idx2word):
    #train net
    load_num = 0
    #net.load_state_dict(torch.load(os.path.join(model_path, "model-%d.pkl" %load_num)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    Dadaset = Poem_DataSet()
    total_loss = 0
    for nowepoch in range(load_num, epoch):
        print("epoch = %d" %(nowepoch + 1))
        for idx, val in enumerate(Dadaset.fetch_data(batch_size=128)):
            X, Y = Variable(torch.from_numpy(np.array(val[0], np.int32)).long()).cuda(), Variable(torch.from_numpy(np.array(val[1])).long()).cuda()
            #print(X.shape)
            out, _ = net.forward(X)
            loss = cross_entropy(out, Y)
            total_loss += loss.cpu().data.numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (idx+1) % 100 == 0:
                print("idx = %d, loss = %lf" %(idx+1, total_loss / 100))
                total_loss = 0
                poem = generate_poem(net, start_words="千山鸟飞绝", word2idx=word2idx, idx2word=idx2word)
                for idx, word in enumerate(poem):
                    poem[idx] = idx2word[poem[idx]]
                print(poem)
        if (nowepoch + 1) % 4 == 0:
            torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" %(nowepoch + 1)))





if __name__ == "__main__":
    data, word2idx, idx2word = get_raw_data()
    net = LSTM_net(voc_size = 8300, embedding_dim = 128, hidden_dim = 256)
    net = net.cuda()
    train_net(net=net, epoch = 256, word2idx=word2idx, idx2word=idx2word)
