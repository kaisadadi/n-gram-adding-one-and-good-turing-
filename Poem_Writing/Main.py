from DataSet import Poem_DataSet, Seq_Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
from DataSet import get_raw_data
import numpy as np
from LSTM_net import LSTM_net, E_D_net, bi_E_D_net
import argparse
import os
from gensim.models import Word2Vec

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", "-g")
parser.add_argument("--task","-t")
#seq代表是序列，jz代表是句子
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
model_path = "/home/wke18/Model/%s" %args.task


def cross_entropy(X, Y):
    #print(X.shape)
    X = X.view(-1, 8300)
    Y = Y.view(-1, 1)
    X = nn.Softmax()(X)
    X = torch.clamp(X, 1e-6, 1 - 1e-6)
    pos = X.topk(10)
    Y_onehot = Variable(torch.zeros(Y.shape[0], 8300).scatter_(1, Y.cpu().data, 1)).cuda()
    loss = torch.sum(-Y_onehot * torch.log(X) - (1 - Y_onehot) * torch.log(1 - X), dim=1).view(-1, 5)
    #loss = torch.sum(-Y_onehot * torch.log(X), dim=1).view(-1, 124)
    return torch.mean(torch.sum(loss, dim=1))

def mse_loss(X, Y, self_embedding):
    seq_size = Y.shape[1]
    #print(seq_size)
    X = X.view(-1, 8300)
    Y = Y.view(-1, 128)
    X = nn.Softmax()(X)
    X = torch.argmax(X, dim=1)
    #print(X)
    #X = F.embedding(X.long(), self_embedding)
    embedding = nn.Embedding.from_pretrained(self_embedding, freeze=False)
    X = embedding(X)
    #print(X)
    #print(nn.MSELoss()(X, Y.double()))
    return nn.MSELoss()(X, Y.double()) * seq_size



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
        pre = []
        for idx, val in enumerate(prefix):
            pre.append(word2idx[val])
        prefix = Variable(torch.from_numpy(np.array(pre)).long().view(1, -1)).cuda()
        _, hidden = net.forward(prefix)
    else:
        hidden = None
    write_sequence = [[]]
    for word in start_words:
        write_sequence[-1].append(word2idx[word])
    while len(write_sequence) < expected_length:
        results, hidden = net.forward(Variable(torch.from_numpy(np.array(write_sequence[-1])).long().view(1, -1)).cuda(), hidden)
        write_sequence.append([])
        results = nn.Softmax()(results.view(-1, 8300))
        for result in results:
            write_sequence[-1].append(int(torch.argmax(result).cpu().data[0].numpy()))
    for idx1, sequence in enumerate(write_sequence):
        for idx2, val in enumerate(sequence):
            write_sequence[idx1][idx2] = idx2word[write_sequence[idx1][idx2]]
    return write_sequence



def train_net(net, epoch, word2idx, idx2word, self_embedding=None):
    #train net
    load_num = 0
    #net.load_state_dict(torch.load(os.path.join(model_path, "model-%d.pkl" %load_num)))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    real_embedding = Variable(torch.from_numpy(np.array(get_real_embedding_matrix(self_embedding, idx2word)))).cuda()
    if args.task == "seq":
        Dataset = Seq_Dataset()
    else:
        Dataset = Poem_DataSet()
    total_loss = 0
    for nowepoch in range(load_num, epoch):
        print("epoch = %d" %(nowepoch + 1))
        for idx, val in enumerate(Dataset.fetch_data(batch_size=64, self_embedding=self_embedding)):
            X, Y = Variable(torch.from_numpy(np.array(val[0]))).cuda(), Variable(torch.from_numpy(np.array(val[1]))).cuda()
            #print(X.shape)
            out, _ = net.forward(X)
            #loss = cross_entropy(out, Y)
            loss = mse_loss(out, Y, real_embedding)
            total_loss += loss.cpu().data.numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (idx+1) % 500 == 0:
                print("idx = %d, loss = %lf" %(idx+1, total_loss / 500))
                total_loss = 0
                #poem = generate_seq(net, start_words="千山鸟飞绝",prefix=None,word2idx=word2idx, idx2word=idx2word)
                #for idx, word in enumerate(poem):
                #    poem[idx] = idx2word[poem[idx]]
                print(poem)
        #if (nowepoch + 1) % 32 == 0:
        #    torch.save(net.state_dict(), os.path.join(model_path, "model-%d.pkl" %(nowepoch + 1)))

def get_real_embedding_matrix(self_embedding, idx2word):
    real_embedding = []
    for a in range(8300):
        try:
            real_embedding.append(self_embedding[idx2word[a]])
        except:
            real_embedding.append(list(np.random.random_sample(size=128)))
    return real_embedding


def eval_net(net, word2idx, idx2word, startword, prefix, load_num):
    net.load_state_dict(torch.load(os.path.join(model_path, "model-%d.pkl" % load_num)))
    poem = generate_seq(net, start_words=startword, prefix=prefix, word2idx=word2idx, idx2word=idx2word)
    print(poem)

if __name__ == "__main__":
    data, word2idx, idx2word = get_raw_data()
    if args.task == "seq":
        net = bi_E_D_net(voc_size = 8300, embedding_dim = 128, hidden_dim = 256)
    else:
        net = LSTM_net(voc_size = 8300, embedding_dim = 128, hidden_dim = 256)
    net = net.cuda()
    import pickle
    matrix = pickle.load(open("Word_Vec.pkl", "rb"))
    #import gensim
    #matrix = gensim.models.KeyedVectors.load_word2vec_format("Word_Vec")
    train_net(net=net, epoch = 1024, word2idx=word2idx, idx2word=idx2word, self_embedding=matrix)
    #eval_net(net, word2idx, idx2word, "不如来饮酒", "相对醉厌厌", load_num=64)
