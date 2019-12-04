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

        if not (h_d.size() == h_e.size()):
            raise Exception("The size of hidden units for encoder and decoder must match!")

        b, seq_len, _ = list(h_d.size())

        Wd_hd = self.W_d(h_d).unsqueeze(2).repeat(1, 1, seq_len, 1)
        We_he = self.W_e(h_e).unsqueeze(1).repeat(1, seq_len, 1, 1)

        e = self.v(torch.tanh(Wd_hd + We_he)).squeeze(-1) # B x t_k x t_k
        a = torch.softmax(e, dim=-1) # B x t_k x t_k
        c = torch.bmm(a, h_e).squeeze(1) # B x t_k x 2*hidden_size

        return c



class VAE(nn.Module):
    def __init__(self, embed_w, embed_size, hidden_size, dropout_p=0.1):
        super(VAE, self).__init__()
        """
        As specified in the paper [https://arxiv.org/pdf/1708.00625.pdf],
        dimension of the latent variable z is equal to the size of hidden units
        """
        self.encoded = False

        latent_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(embed_w)
        self.dropout = nn.Dropout(self.dropout_p)

        # Encoder
        self.encoder = nn.GRU(emb_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.attention = SimpleAttention(hidden_size)

        # Decoder (deterministic)
        self.decoder_1 = nn.GRU(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.decoder_2 = nn.GRU(embed_size + 2 * hidden_size, hidden_size, num_layers=1, batch_first=True)

        # VAE Encoder
        self.W_yh_ez = nn.Linear(embed_size, hidden_size)
        self.W_zh_ez = nn.Linear(latent_size, hidden_size)
        self.W_hh_ez = nn.Linear(hidden_size, hidden_size)

        self.W_mu = nn.Linear(hidden_size, latent_size)
        self.W_logvar = nn.Linear(hidden_size, latent_size)

        # VAE Decoder
        self.W_zh_dy = nn.Linear(latent_size, hidden_size)
        self.W_hh_dy = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, emb_size)



        self.n_z = opt.n_z

    def encode(self, x):
        x = self.embedding(x) # B x t_k x embedding_size
        x = self.dropout(x)

        self.h_e, h_n = encoder(x) # B x t_k x 2*hidden_size

        self.encoded = True


    def decode(self):

        h_0 = torch.mean(self.h_e, 1).unsqueeze(0)

        self.h_d1, _ = self.decoder_1(self.y, h_0) # B x t_k x hidden_size

        self.c = self.attention(self.h_d_1, self.h_e) # B x t_k x 2*hidden_size

        self.h_d_2, _ = self.decoder_2(torch.cat((self.c, self.y),
                                                  dim=-1))  # B x t_k x hidden_size


    def generate(self, t, z=None):

        if z == None:
            # Since latent z has same dimension as hidden state h_d
            z = torch.zeros(self.h_d_2[:, 0].size())

        self.h_t_ez = torch.sigmoid(
            self.W_yh_ez(self.y[:, t]) + \
            self.W_zh_ez(z) + \
            self.W_hh_ez(self.h_d_1[:, t]))

        mean_t = self.W_mu(h_t_ez)
        sigma_t = torch.sqrt(torch.exp(self.W_logvar(h_t_ez)))

        eps = torch.randn(mean_t.size())

        z_t = mean_t + sigma * eps

        h_dy = torch.tanh(
            self.W_zh_dy(z_t) + \
            self.W_hh_dy(h_d_2[:, t]))

        y_t = torch.softmax(self.W_hy(h_dy), dim=-1)

        return 





    def forward(self, y, h_d1=None, h_d2, z=None):
        """
        Generate output for the VAE decoder for a single timestep
        """

        if not self.encoded:
            raise Exception("Need to first encode input sequence.")



        # y = self.embedding(x) # B x t_k x embedding_size

        # h_e, _ = encoder(y) # B x t_k x 2*hidden_size

        # h_d_1, _ = self.decoder_1(y) # B x t_k x hidden_size

        # c = self.attention(h_d, h_e) # B x t_k x 2*hidden_size
        
        # h_d_2, _ = self.decoder_2(torch.cat((c, y), dim=-1))  # B x t_k x hidden_size

        self.encode(x)
        self.decode()

        # Generator










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