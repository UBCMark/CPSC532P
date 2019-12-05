import torch
import torch.nn as nn
import pdb
from data import cfg
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-cfg.RAND_UNIF_INIT_MAG, cfg.RAND_UNIF_INIT_MAG)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=cfg.TRUNC_NORM_STD)
    if linear.bias is not None:
        linear.bias.data.normal_(std=cfg.TRUNC_NORM_STD)


class EncoderRNN(nn.Module):
    def __init__(self, weights, emb_size, hidden_size, n_layers):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(weights)
        if cfg.LSTM:
            self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=True)
        else:
            self.gru = nn.GRU(emb_size, hidden_size, n_layers, bidirectional=True)

    def forward(self, input, hidden):
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


class EncoderRNN_batch(nn.Module):
    def __init__(self, weights, emb_size, hidden_size, n_layers):
        super(EncoderRNN_batch, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        if cfg.LSTM:
            self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, bidirectional=True)
            init_lstm_wt(self.rnn)
        else:
            self.rnn = nn.GRU(emb_size, hidden_size, n_layers, batch_first=True, bidirectional=True)

    # seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
        output, hidden = self.rnn(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()

        return encoder_outputs, hidden

    def initHidden(self, B, device):
        if cfg.LSTM:
            return (torch.zeros(B, self.n_layers * 2, self.hidden_size, device=device),
                    torch.zeros(B, self.n_layers * 2, self.hidden_size, device=device))
        else:
            return torch.zeros(B, 2 * self.n_layers, self.hidden_size, device=device)