import torch
import torch.nn as nn
from torch import optim
from data.dataset import SummarizationDataset, get_dataloader
from models.VAE import VAE

from data import cfg
import os, sys, time, math, random
import pdb
from utils.model_saver_iter import load_model, save_model
from torch.autograd import Variable
import numpy as np

MAX_LENGTH = 500

def trainVAE(model, dataloader, learning_rate=0.005, tf_ratio=0.5, print_every=1000):
    start = time.time()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    batch_ll_loss, batch_kl_loss = 0., 0.

    for iter, (input_idx, input_mask, target_idx, target_mask) in enumerate(dataloader):
        input_idx = Variable(input_idx).to(device)
        target_idx = Variable(target_idx).to(device)
        input_mask = input_mask.to(device)
        

        batch_size = input_idx.size(0)

        input_len = input_mask.sum(dim=-1)
        target_len = target_mask.sum(dim=-1)

        ll_loss, kl_loss = 0., 0
        optimizer.zero_grad()

        model.encode(input_idx, input_mask)
        pdb.set_trace()

        # EOS token as the first input
        y0 = Variable(torch.LongTensor([[200001]])).to(device)

        y, h_d1, h_d2, z, kl = model(y0)

        ll_loss += F.nll_loss(y, target_idx[:, 0])
        kl_loss += kl

        for i in range(target_length):

            # Use ground-truth token as input
            if random.random() < tf_ratio:
                y = target_idx[:, i].unsqueeze(1)

            # Use predicted token as input
            else:
                y = y.topk(1)[1].detach()

            if y.item() == 200001:
                break

            y, h_d1, h_d2, z, kl = model(y, h_d1, h_d2, z)

            ll_loss += criterion(y, batch_target[:, 0])
            kl_loss += kl

        loss = ll_loss + kl_loss
        loss.backward()
        optimizer.step()

        if (iter + 1) % print_every == 0:
            print("Likelihood loss: {}, KL loss: {}".format(ll_loss.item(), kl_loss.item()))

        batch_ll_loss += ll_loss.item() / target_length
        batch_kl_loss += kl_loss.item() / target_length


    return model, batch_ll_loss, batch_kl_loss


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python train.py <checkpoint_dir>")
        sys.exit()
    checkpoint_dir = sys.argv[1]

    device = torch.device('cuda:0')
    hidden_size = 256
    output_size = 200003
    weights = torch.load("data/GloVe_embeddings.pt")

    model = VAE(weights, cfg.EMBEDDING_SIZE, hidden_size, output_size)
    model.to(device)
    model.init_weights()

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    dataloader = get_dataloader(SummarizationDataset("data/finished/val.txt", "data/word2idx.json"))

    n_epochs = 105

    # Starts decaying teacher-forcing ratio at this epoch
    tf_decay_start = 5

    for epoch in range(1, n_epochs + 1):

        tf_ratio = 1 - max(0, epoch - tf_decay_start) / (n_epochs - tf_decay_start)

        model, ll_loss, kl_loss = trainVAE(model, dataloader, tf_ratio=tf_ratio, print_every=1000)

        with open('loss.txt', 'a') as f:
            f.write("[Epoch {}] Likelihood Loss: {.:6f}, KL Loss: {.:6f}\n".format(epoch, ll_loss, kl_loss))

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "{}.pth".format(epoch)))