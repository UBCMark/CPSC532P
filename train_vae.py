import torch
import torch.nn as nn
import torch.nn.functional as F
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

def trainVAE(model, dataloader, lr=0.0005, tf_ratio=0.5, print_every=1000):
    start = time.time()

    optimizer = optim.SGD(model.parameters(), lr=lr)

    batch_nll_loss, batch_kl_loss = 0., 0.

    for iter, (input_idx, input_mask, target_idx, target_mask) in enumerate(dataloader):
        start = time.time()

        # Move input tensors to GPU if available
        input_idx, target_idx = input_idx.to(device), target_idx.to(device)
        input_mask, target_mask = input_mask.to(device), target_mask.to(device)

        batch_size = input_idx.size(0)

        input_len = input_mask.sum(dim=-1)
        target_len = target_mask.sum(dim=-1)

        nll_loss, kl_loss = 0., 0
        optimizer.zero_grad()

        model.encode(input_idx, input_mask)

        # EOS token as the first input
        y0 = torch.LongTensor([[200001]]).repeat(batch_size, 1).to(device)

        # Compute loss at time-step 0
        y, h_d1, h_d2, z, kl = model(y0)
        nll_loss += F.nll_loss(y, target_idx[:, 0])
        kl_loss += kl.sum()

        for t in range(1, target_idx.size(1)):

            # The target mask for mini-batch as time-step t
            target_mask_t = target_mask[:, t]

            if target_mask_t.sum().item() == 0:
                break

            # Use ground-truth token as input
            if random.random() < tf_ratio:
                y = target_idx[:, t - 1].unsqueeze(1)

            # Use predicted token as input
            else:
                y = y.topk(1)[1].detach()

            y, h_d1, h_d2, z, kl = model(y, h_d1, h_d2, z)

            nll_loss += F.nll_loss(y.masked_fill(target_mask_t.unsqueeze(1) == 0, 0),
                                   target_idx[:, t].masked_fill(target_mask_t == 0, 0))
            kl_loss += kl.masked_fill(target_mask_t == 0, 0).sum()

        loss = (nll_loss + kl_loss) / batch_size
        loss.backward()
        optimizer.step()

        end = time.time()

        if (iter + 1) % print_every == 0:
            print("[{} of {}] NLL Loss: {:.6f}, KL Loss: {:.6f} {:.3f} sec)"
                .format(iter + 1,
                        len(dataloader),
                        nll_loss.item() / batch_size,
                        kl_loss.item() / batch_size,
                        end - start))
    return model


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

    dataset = SummarizationDataset("data/finished/train.txt", "data/word2idx.json")
    dataloader = get_dataloader(dataset, batch_size=14)

    n_epochs = 105

    # Starts decaying teacher-forcing ratio at this epoch
    tf_decay_start = 5

    for epoch in range(1, n_epochs + 1):

        tf_ratio = 1 - max(0, epoch - tf_decay_start) / (n_epochs - tf_decay_start)

        if epoch == 1:
            lr = 0.0005
        else:
            lr = 0.001

        model, ll_loss, kl_loss = trainVAE(model, dataloader, lr=lr, tf_ratio=tf_ratio, print_every=10)

        with open('loss.txt', 'a') as f:
            f.write("[Epoch {}] Likelihood Loss: {.:6f}, KL Loss: {.:6f}\n".format(epoch, ll_loss, kl_loss))

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "{}.pth".format(epoch)))