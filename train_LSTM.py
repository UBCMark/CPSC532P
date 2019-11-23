import torch
import torch.nn as nn
from torch import optim
from data.dataset import SummarizationDataset
from data.dataset import get_dataloader
from models.AttnDecoderRNN import AttnDecoderRNN
from models.EncoderRNN import EncoderRNN
from data import cfg
import os, sys, time, math, random
import pdb

MAX_LENGTH = 500
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size * 2, device=device)

    loss = 0
    for ei in range(input_length):

        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        
        try:
            encoder_outputs[ei] = encoder_output[0, 0]
        except:
            pdb.set_trace()

    decoder_input = torch.tensor([[20000]], device=device)

    h, c = encoder_hidden
    h = torch.cat((h[0], h[1]), 1).unsqueeze(0)
    c = torch.cat((c[0], c[1]), 1).unsqueeze(0)
    decoder_hidden = (h, c)

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


def trainIters(encoder, decoder, checkpoint_dir, print_every=10, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    dataloader = get_dataloader(SummarizationDataset("data/finished/train.txt", "data/word2idx.json"))

    criterion = nn.NLLLoss()

    for iter, batch in enumerate(dataloader):
        input_tensor = batch[0][0].to(device)
        target_tensor = batch[1][0].to(device)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss


        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / len(dataloader) * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    return encoder, decoder, plot_loss_total


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python train.py <checkpoint_dir>")
        sys.exit()
    checkpoint_dir = sys.argv[1]

    device = torch.device('cuda:0')
    hidden_size = 256
    weights = torch.load("data/GloVe_embeddings.pt")
    encoder = EncoderRNN(weights, cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, 1).to(device)
    decoder = AttnDecoderRNN(weights, cfg.HIDDEN_SIZE * 2, 200003, 1, dropout_p=0.1).to(device)

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    for epoch in range(120):
        encoder, decoder, loss = trainIters(encoder, decoder, checkpoint_dir, print_every=1000)


        # print("Epoch {}: {.:3f}".format(epoch, loss))
        # with open('loss.txt', 'a') as f:
        #     f.write("Epoch {}: {}\n".format(epoch, loss))

        torch.save(encoder.state_dict(), os.path.join(checkpoint_dir, "encoder_{}.pth".format(epoch + 1)))
        torch.save(decoder.state_dict(), os.path.join(checkpoint_dir, "decoder_{}.pth".format(epoch + 1)))
