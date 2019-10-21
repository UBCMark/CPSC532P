import torch
import torch.nn as nn
import pdb

class EncoderRNN(nn.Module):
    def __init__(self, weights, emb_size, hidden_size, n_layers, dropout_p):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, dropout=dropout_p, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(2*self.n_layers, 1, self.hidden_size, device=device)
