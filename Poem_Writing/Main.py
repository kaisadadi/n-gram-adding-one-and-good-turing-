from Poem_Writing.DataSet import Poem_DataSet
from torch.autograd import Variable
import torch.nn as nn
import torch
from Poem_Writing.DataSet import get_raw_data
import numpy as np
from Poem_Writing.LSTM_net import LSTM_net


def cross_entropy(X, Y):
    print(X.shape)
    X = X.view(-1, 8300)
    Y = Y.view(-1, 1)
    #Y = torch.clamp(Y, max=5)
    X = nn.Softmax()(X)
    Y_onehot = Variable(torch.zeros(Y.shape[0], 8300).scatter_(1, Y.data, 1))
    loss = torch.sum(-Y_onehot * torch.log(X) - (1 - Y_onehot) * torch.log(1 - X), dim=1).view(32, 124)
    return torch.mean(torch.sum(loss, dim=1))

def generate_poem(net, start_words, word2idx, idx2word, expected_length = 26):
    #为你写诗，为你静止
    #26代表经典的五言诗(4句)
    write_sequence = [word2idx["<START>"]]
    for word in start_words:
        write_sequence.append(word2idx[word])
    while len(write_sequence) < expected_length:
        results, _ = net.forward(Variable(torch.from_numpy(np.array(write_sequence)).long().view(1, -1)))
        results = results.view(-1, 8300)
        result = results[-1].view(-1)
        prob = nn.Softmax()(result)
        print(prob.shape)
        print(prob)
        print(torch.max(prob, dim=0))
        write_sequence.append(idx2word[torch.argmax(prob)[1]])
    return write_sequence

def train_net(net, epoch, word2idx, idx2word):
    #train net
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Dadaset = Poem_DataSet()
    total_loss = 0
    for nowepoch in range(epoch):
        for idx, val in enumerate(Dadaset.fetch_data(batch_size=32)):
            X, Y = Variable(torch.from_numpy(np.array(val[0], np.int32)).long()), Variable(torch.from_numpy(np.array(val[1])).long())
            print(X.shape)
            out, _ = net.forward(X)
            loss = cross_entropy(out, Y)
            total_loss += loss.data.numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 2 == 0:
                print("idx = %d, loss = %lf" %(idx, total_loss / 100))
                total_loss = 0
                generate_poem(net, start_words="深度学习好", word2idx=word2idx, idx2word=idx2word)





if __name__ == "__main__":
    data, word2idx, idx2word = get_raw_data()
    net = LSTM_net(voc_size = 8300, embedding_dim = 128, hidden_dim = 256)
    train_net(net=net, epoch = 32, word2idx=word2idx, idx2word=idx2word)