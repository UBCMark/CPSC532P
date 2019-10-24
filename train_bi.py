import torch
import torch.nn as nn
from torch import optim
from data.dataset import SummarizationDataset
from data.dataset import get_dataloader
from models.AttnDecoderRNN_bi import AttnDecoderRNN
from models.EncoderRNN_bi import EncoderRNN
from data import cfg
import os, sys, time, math, random
import pdb
from utils.model_saver_iter import load_model, save_model

MAX_LENGTH = 500
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, 2*encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        try:
            encoder_outputs[ei] = encoder_output[0, 0]
            pdb.set_trace()
        except:
            pdb.set_trace()

    decoder_input = torch.tensor([[20000]], device=device)

    decoder_hidden = encoder_hidden.view(1,1,-1)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di].view(-1))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].view(-1))
            if decoder_input.item() == 200001:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, checkpoint_dir, print_every=1000, plot_every=100, learning_rate=0.01,
               save_every=1000):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    dataloader = get_dataloader(SummarizationDataset("data/finished/train.txt", "data/word2idx.json"))

    criterion = nn.NLLLoss()
    start_iter = load_model(encoder, model_dir=checkpoint_dir, appendix='Encoder', iter="l")
    start_iter_ = load_model(decoder, model_dir=checkpoint_dir, appendix='Decoder', iter="l")
    assert start_iter == start_iter_

    data_iter = iter(dataloader)

    if start_iter < n_iters:

        for i in range(start_iter, n_iters):
            try:
                batch = next(data_iter)
            except:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_tensor = batch[0][0].to(device)
            target_tensor = batch[1][0].to(device)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('(%d %d%%) %.4f' % (i, i / n_iters * 100, print_loss_avg))

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            # Save checkpoint
            # torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, "encoder_{}.pth".format(iter)))
            # torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, "decoder_{}.pth".format(iter)))
            if (i + 1) % save_every == 0:
                save_model(encoder, model_dir=checkpoint_dir, appendix="Encoder", iter=i + 1, save_num=3,
                           save_step=save_every)
                save_model(decoder, model_dir=checkpoint_dir, appendix="Decoder", iter=i + 1, save_num=3,
                           save_step=save_every)
        # showPlot(plot_losses)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python train.py <checkpoint_dir>")
        sys.exit()
    checkpoint_dir = sys.argv[1]

    device = torch.device('cuda:1')
    hidden_size = 256
    weights = torch.load("data/GloVe_embeddings.pt")
    encoder1 = EncoderRNN(weights, cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, 1, dropout_p=0.1).to(device)
    attn_decoder1 = AttnDecoderRNN(weights, 2*cfg.HIDDEN_SIZE, 200003, 1, dropout_p=0.1).to(device)

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    trainIters(encoder1, attn_decoder1, 200000, checkpoint_dir, print_every=10, save_every=100)
