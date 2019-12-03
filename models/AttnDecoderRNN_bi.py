import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from data import cfg

MAX_LENGTH = 500


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


class AttnDecoderRNN(nn.Module):
    def __init__(self, weights, hidden_size, output_size, num_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.num_layers = num_layers

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.attn = nn.Linear(self.hidden_size + 100, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + 100, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_hiddens):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # TODO: recalculate weight, include coverage
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_hiddens.unsqueeze(0))

        # c_t and y_t-1
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights



    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=device)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if cfg.IS_COVERAGE:
            self.W_c = nn.Linear(1, cfg.HIDDEN_SIZE * 2, bias=False)
        if cfg.LSTM:
            self.decode_proj = nn.Linear(cfg.HIDDEN_SIZE * 2, cfg.HIDDEN_SIZE * 2)
        else:
            self.decode_proj = nn.Linear(cfg.HIDDEN_SIZE, cfg.HIDDEN_SIZE * 2)
            
        self.v = nn.Linear(cfg.HIDDEN_SIZE * 2, 1, bias=False)

        self.W_h = nn.Linear(cfg.HIDDEN_SIZE * 2, cfg.HIDDEN_SIZE * 2, bias=False)

    def forward(self, decoder_hidden_hat, encoder_hiddens, coverage):
        b, t_k, n = list(encoder_hiddens.size())

        encoder_feature = encoder_hiddens.view(-1, 2 * cfg.HIDDEN_SIZE)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        dec_fea = self.decode_proj(decoder_hidden_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if cfg.IS_COVERAGE:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) # * enc_padding_mask  # B x t_k
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_hiddens)  # B x 1 x n
        c_t = c_t.view(-1, cfg.HIDDEN_SIZE * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if cfg.IS_COVERAGE:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(cfg.HIDDEN_SIZE * 2, cfg.HIDDEN_SIZE)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(cfg.HIDDEN_SIZE* 2, cfg.HIDDEN_SIZE)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        if cfg.LSTM:
            h, c = hidden  # h, c dim = 2 x b x hidden_dim
            h_in = h.transpose(0, 1).contiguous().view(-1, cfg.HIDDEN_SIZE * 2)
            hidden_reduced_h = F.relu(self.reduce_h(h_in))
            c_in = c.transpose(0, 1).contiguous().view(-1, cfg.HIDDEN_SIZE * 2)
            hidden_reduced_c = F.relu(self.reduce_c(c_in))
            return hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)
        else:
            h = hidden
            h_in = h.transpose(0, 1).contiguous().view(-1, cfg.HIDDEN_SIZE * 2)
            hidden_reduced_h = F.relu(self.reduce_h(h_in))
            return hidden_reduced_h.unsqueeze(0)# , hidden_reduced_c.unsqueeze(0) # h, c dim = 1 x b x hidden_dim


class AttnDecoderRNN_full(nn.Module):
    def __init__(self, weights):
        super(AttnDecoderRNN_full, self).__init__()
        self.reduce_state = ReduceState()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding.from_pretrained(weights)
        # init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(cfg.HIDDEN_SIZE * 2 + cfg.EMBEDDING_SIZE, cfg.EMBEDDING_SIZE)

        if cfg.LSTM:
            self.rnn = nn.LSTM(cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, num_layers=1, batch_first=True)
            init_lstm_wt(self.rnn)
            if cfg.POINTER_GEN:
                self.p_gen_linear = nn.Linear(cfg.HIDDEN_SIZE * 4 + cfg.EMBEDDING_SIZE, 1)  # was 4 if with c
        else:
            self.rnn = nn.GRU(cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, num_layers=1)
            if cfg.POINTER_GEN:
                self.p_gen_linear = nn.Linear(cfg.HIDDEN_SIZE * 3 + cfg.EMBEDDING_SIZE, 1)  # was 4 if with c

        # p_vocab
        self.out1 = nn.Linear(cfg.HIDDEN_SIZE * 3, cfg.HIDDEN_SIZE)
        self.out2 = nn.Linear(cfg.HIDDEN_SIZE, cfg.VOCAB_SIZE+3)
        # init_linear_wt(self.out2)

    def forward(self, encoder_hiddens, decoder_input, decoder_hidden,
                c_t_1, coverage, input_idx, step):

        if not self.training and step == 0:
            if cfg.LSTM:
                h_decoder, c_decoder = decoder_hidden
                decoder_hidden_hat = torch.cat((h_decoder.view(-1, cfg.HIDDEN_SIZE),
                                     c_decoder.view(-1, cfg.HIDDEN_SIZE)), 1)  # B x 2*hidden_dim
            else:
                decoder_hidden_hat = decoder_hidden

            c_t, _, coverage_next = self.attention_network(decoder_hidden_hat, encoder_hiddens, coverage)

            coverage = coverage_next

        embd = self.embedding(decoder_input)
        x = self.x_context(torch.cat((c_t_1, embd.view(1,-1)), 1))
        decoder_output, decoder_hidden = self.rnn(x.unsqueeze(1), decoder_hidden)

        if cfg.LSTM:
            h_decoder, c_decoder = decoder_hidden
            decoder_hidden_hat = torch.cat((h_decoder.view(-1, cfg.HIDDEN_SIZE),
                                 c_decoder.view(-1, cfg.HIDDEN_SIZE)), 1)  # B x 2*hidden_dim
        else:
            h_decoder = decoder_hidden
            decoder_hidden_hat = h_decoder.view(-1, cfg.HIDDEN_SIZE)

        c_t, attn_dist, coverage_next = self.attention_network(decoder_hidden_hat, encoder_hiddens, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if cfg.POINTER_GEN:
            p_gen_input = torch.cat((c_t, decoder_hidden_hat, x), 1)  # lstm: B x (2*2*hidden_dim + emb_dim) gru: B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((decoder_output.view(-1, cfg.HIDDEN_SIZE), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if cfg.POINTER_GEN:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            final_dist = vocab_dist_.scatter_add(1, input_idx.view(1,-1), attn_dist_)
            # final_dist = vocab_dist_ + attn_dist_
        else:
            final_dist = vocab_dist

        return final_dist, decoder_hidden, c_t, attn_dist, p_gen, coverage
