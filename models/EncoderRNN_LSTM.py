import torch
import torch.nn as nn
import pdb


class EncoderRNN(nn.Module):
    def __init__(self, weights, emb_size, hidden_size, n_layers):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self, device):
        return (torch.zeros(self.n_layers * 2, 1, self.hidden_size, device=device),
                torch.zeros(self.n_layers * 2, 1, self.hidden_size, device=device))