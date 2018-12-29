import torch.nn as nn
import torch
from torch.autograd import Variable


class E_D_net(nn.Module):
    def __init__(self, voc_size, embedding_dim, hidden_dim):
        #an LSTM based encoder-decoder framework
        super(E_D_net, self).__init__()
        self.embedding = nn.Embedding(voc_size, embedding_dim)
        self.hidden_size = hidden_dim
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2)
        self.fc = nn.Linear(hidden_dim, voc_size)

    def forward(self, x, hidden_state=None):
        embed = self.embedding(x)
        embed = torch.transpose(embed, 0, 1)
        if hidden_state == None:
            hidden_state = (Variable(torch.zeros(2, embed.shape[1], self.hidden_size)), Variable(torch.zeros(2, embed.shape[1], self.hidden_size)))
        _, (encoder_result_h, encoder_result_c) = self.encoder(embed, hidden_state)
        zero_input = Variable(torch.zeros(embed.shape[0], embed.shape[1], embed.shape[2]))  #seqlen bs input_size
        decoder_result, _ = self.decoder(zero_input, (encoder_result_h, encoder_result_c))
        out_result = torch.transpose(decoder_result, 0, 1)  #bs seqlen hidden_size
        out_result = self.fc(out_result)
        return out_result

class LSTM_net(nn.Module):
    def __init__(self, voc_size, embedding_dim, hidden_dim):
        #an LSTM based encoder-decoder framework
        super(LSTM_net, self).__init__()
        self.embedding = nn.Embedding(voc_size, embedding_dim)
        self.hidden_size = hidden_dim
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2)
        self.fc = nn.Linear(hidden_dim, voc_size)

    def forward(self, x, hidden_state=None):
        embed = self.embedding(x)
        embed = torch.transpose(embed, 0, 1)
        if hidden_state == None:
            hidden_state = (Variable(torch.zeros(2, embed.shape[1], self.hidden_size)), Variable(torch.zeros(2, embed.shape[1], self.hidden_size)))
        out_result, hidden = self.LSTM(embed, hidden_state)
        out_result = torch.transpose(out_result, 0, 1)  #bs seqlen hidden_size
        out_result = self.fc(out_result)
        return out_result, hidden



if __name__ == "__main__":
    net = LSTM_net(voc_size=10, embedding_dim=3, hidden_dim=15)
    import numpy as np
    net.forward(Variable(torch.from_numpy(np.array([[1, 2, 3 ,4], [5, 6, 7, 8]], dtype=np.int32)).long()))