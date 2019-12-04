import torch
import torch.nn as nn
import pdb
from data import cfg


class EncoderRNN(nn.Module):
    def __init__(self, weights, emb_size, hidden_size, n_layers, dropout_p):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(weights)
        if cfg.LSTM:
            self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=True, batch_first=True)
        else:
            self.gru = nn.GRU(emb_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=True, batch_first=True)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input).view(1, 1, -1)
        if cfg.LSTM:
            output, hidden = self.lstm(embedded, hidden)
        else:
            output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self, device):
        if cfg.LSTM:
            return (torch.zeros(self.n_layers * 2, 1, self.hidden_size, device=device),
                    torch.zeros(self.n_layers * 2, 1, self.hidden_size, device=device))
        else:
            return torch.zeros(2 * self.n_layers, 1, self.hidden_size, device=device)
