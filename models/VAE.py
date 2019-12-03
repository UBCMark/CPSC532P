import torch as T
import torch.nn as nn
import torch.nn.functional as F

from EncoderRNN_bi import EncoderRNN



class SimpleAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleAttention, self).__init__()

        self.W_d = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_e = nn.Linear(hidden_size * 2, hidden_size, bias=True)

        self.v = nn.Linear(hidden_size, 1, bias=True)


    def forward(self, h_d, h_e):

        b, seq_len, _ = list(h_d.size())

        e_t = self.v(torch.tanh(self.W_d(h_d) + self.W_e(h_e))).squeeze(-1) # B x t_k
        a_t = torch.softmax(e_t, dim=-1).unsqueeze(1) # B x 1 x t_k
        c_t = torch.bmm(a_t, h_e).squeeze(1) # B x 2*hidden_size

        return c_t



class VAE(nn.Module):
    def __init__(self, embed_w, embed_size, hidden_size, dropout_p=0.1):
        super(VAE, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embed_w)
        self.dropout = nn.Dropout(self.dropout_p)

        self.encoder = EncoderRNN(embed_w, embed_size, hidden_size)
        self.rnn_1 = nn.GRU(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.attention = SimpleAttention(hidden_size)

        self.hidden_to_mu = nn.Linear(2*hidden_size, opt.n_z)
        self.hidden_to_logvar = nn.Linear(2*hidden_size, opt.n_z)
        self.n_z = opt.n_z

    def forward(self, x, G_inp, z = None, G_hidden = None):

        h_e, _ = encoder(x)

        h_d, _ = self.rnn1()

        embeddings = self.embedding(x)
        embedding = self.dropout(embeddings)

        c_t = self.attention(h_d, h_e)
        





        if z is None:	                                                #If we are testing with z sampled from random noise
            batch_size, n_seq = x.size()
            x = self.embedding(x)	                                    #Produce embeddings from encoder input
            E_hidden = self.encoder(x)	                                #Get h_T of Encoder
            mu = self.hidden_to_mu(E_hidden)	                        #Get mean of lantent z
            logvar = self.hidden_to_logvar(E_hidden)	                #Get log variance of latent z
            z = get_cuda(T.randn([batch_size, self.n_z]))	                #Noise sampled from ε ~ Normal(0,1)
            z = mu + z*T.exp(0.5*logvar)	                            #Reparameterization trick: Sample z = μ + ε*σ for backpropogation
            kld = -0.5*T.sum(logvar-mu.pow(2)-logvar.exp()+1, 1).mean()	#Compute KL divergence loss
        else:
            kld = None                                                  #If we are training with given text

        G_inp = self.embedding(G_inp)	                                #Produce embeddings for generator input

        logit, G_hidden = self.generator(G_inp, z, G_hidden)
        return logit, G_hidden, kld