import torch as torch
import torch.nn as nn
import torch.nn.functional as F

import pdb



class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()

        self.W_d = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_e = nn.Linear(hidden_size * 2, hidden_size, bias=True)

        self.v = nn.Linear(hidden_size, 1, bias=True)


    def forward(self, h_d, h_e):
        b, seq_len, _ = list(h_e.size())

        Wd_hd = self.W_d(h_d.repeat(1, seq_len, 1))         # B x T_e x hidden_size
        We_he = self.W_e(h_e)                               # B x T_e x hidden_size

        e_t = self.v(torch.tanh(Wd_hd + We_he)).squeeze(-1) # B x T_e
        a_t = torch.softmax(e_t, dim=-1)                    # B x T_e
        c_t = torch.bmm(a_t.unsqueeze(1), h_e).squeeze(1)   # B x 2*hidden_size

        return c_t



class VAE(nn.Module):
    def __init__(self, embed_w, embed_size, hidden_size, output_size, dropout_p=0.1):
        super(VAE, self).__init__()
        """
        As specified in the paper [https://arxiv.org/pdf/1708.00625.pdf],
        dimension of the latent variable z is equal to the size of hidden units
        """
        self.encoded = False

        latent_size = hidden_size

        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.embedding = nn.Embedding.from_pretrained(embed_w)
        self.dropout = nn.Dropout(dropout_p)

        # Encoder
        self.encoder_rnn = nn.GRU(embed_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.attention = SimpleAttention(hidden_size)

        # Decoder (deterministic)
        self.decoder_rnn1 = nn.GRU(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.decoder_rnn2 = nn.GRU(embed_size + 2 * hidden_size, hidden_size, num_layers=1, batch_first=True)

        # VAE Encoder
        self.W_yh_ez = nn.Linear(embed_size, hidden_size)
        self.W_zh_ez = nn.Linear(latent_size, hidden_size)
        self.W_hh_ez = nn.Linear(hidden_size, hidden_size)

        self.W_mu = nn.Linear(hidden_size, latent_size)
        self.W_logvar = nn.Linear(hidden_size, latent_size)

        # VAE Decoder
        self.W_zh_dy = nn.Linear(latent_size, hidden_size)
        self.W_hh_dy = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)

    def encode(self, x, mask):
        self.mask = mask

        x = self.embedding(x)          # B x t_k x embedding_size
        x = self.dropout(x)
        self.h_e, h_n = self.encoder_rnn(x) # B x t_k x 2*hidden_size

        self.h_e * self.mask.unsqueeze(-1)
        pdb.set_trace()

        self.encoded = True


    def forward(self, y, h_d1=None, h_d2=None, z=None):
        """
        Generate output for the recurrent VAE decoder for a single timestep
        """
        if not self.encoded:
            raise Exception("Need to first encode input sequence!")

        y = self.embedding(y).squeeze(1)                                                    # B x 1 x embed_size

        # Initialized h_d_0 to be the average of all the encoder input states
        if h_d1 is None or h_d2 is None:
            pdb.set_trace()
            h_d_0 = torch.add(
                torch.mean(self.h_e[:, :, :self.hidden_size], 1),
                torch.mean(self.h_e[:, :, self.hidden_size:], 1)) / 2.
            h_d_0 = h_d_0.detach()                                                          # B x hidden_size

            h_d1 = h_d_0 if h_d1 is None else h_d1
            h_d2 = h_d_0 if h_d2 is None else h_d2

        # Compute the deterministic hidden states of the decoder 
        output, h_d1_t = self.decoder_rnn1(y.unsqueeze(1), h_d1.unsqueeze(1))               # B x 1 x hidden_size

        c_t = self.attention(h_d1_t, self.h_e)                                              # B x 2*hidden_size

        output, h_d2_t = self.decoder_rnn2(torch.cat((y, c_t), dim=-1).unsqueeze(1),
                                           h_d2.unsqueeze(1))                               

        h_d1_t = h_d1_t.squeeze(1)                                                          # B x hidden_size
        h_d2_t = h_d2_t.squeeze(1)                                                          # B x hidden_size

        # Compute latent vector z at current time-step using VAE encoder
        
        if z is None:
            h_ez_t = torch.sigmoid(self.W_yh_ez(y) + self.W_hh_ez(h_d1))                    # B x hidden_size

        # Use latent vector z for VAE hidden state if passed in from previous time-step
        else:
            h_ez_t = torch.sigmoid(self.W_yh_ez(y) + self.W_zh_ez(z) + self.W_hh_ez(h_d1))  # B x hidden_size

        mu_t = self.W_mu(h_ez_t)                                                            # B x latent_size
        logvar_t = self.W_logvar(h_ez_t)                                                    # B x latent_size
        sigma_t = torch.sqrt(torch.exp(logvar_t))                                           # B x latent_size
        eps = torch.randn(mu_t.size())                                                      # B x latent_size

        z_t = mu_t + sigma_t * eps.to(sigma_t.device)                                       # B x latent_size

        # Compute output vector y from latent vector z using VAE decoder
        h_dy_t = torch.tanh(self.W_zh_dy(z_t) + self.W_hh_dy(h_d2_t))                       # B x hidden_size
        y_t = torch.softmax(self.W_hy(h_dy_t), -1)                                          # B x output_size


        # Compute the KL-Divergence between q(z_t|y, z) and p(z)
        KL = 0.5 * torch.sum(logvar_t.exp() + mu_t.pow(2) - logvar_t - 1, 1)

        return y_t, h_d1_t, h_d2_t, z_t, KL


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)