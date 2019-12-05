import torch
import torch.nn as nn
from torch import optim
from data.dataset import SummarizationDataset
from data.dataset import get_dataloader
from models.AttnDecoderRNN_bi import AttnDecoderRNN, AttnDecoderRNN_full, Attention, ReduceState
from models.EncoderRNN_bi import EncoderRNN_batch as EncoderRNN
from data import cfg
import os, sys, time, math, random
import pdb
from utils.model_saver_iter import load_model, save_model
from torch.autograd import Variable
import numpy as np

MAX_LENGTH = 500
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_hiddens = torch.zeros(max_length, 2*encoder.hidden_size, device=device)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        try:
            # here just use the hidden states of 1st dimension,
            # should be justified later
            encoder_hiddens[ei] = encoder_hidden[0, 0]
        except:
            pdb.set_trace()

    decoder_input = torch.tensor([[20000]], device=device)

    decoder_hidden = encoder_hidden.view(1,1,-1)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_hiddens)
            loss += criterion(decoder_output, target_tensor[di].view(-1))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_hiddens)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].view(-1))
            if decoder_input.item() == 200001:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def train_full(input_tensor, input_mask, target_tensor, target_mask, encoder, decoder, encoder_optimizer, decoder_optimizer):
    B = input_tensor.shape[0]
    encoder_hidden = encoder.initHidden(B, device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    max_input_length = int(max(input_mask.sum(1)))
    max_target_length = int(max(target_mask.sum(1)))

    input_mask = input_mask[:, :max_input_length]
    input_idx = input_tensor[:, :max_input_length]

    c_t_1 = Variable(torch.zeros((B, 2 * cfg.HIDDEN_SIZE))).to(device)

    coverage = None
    if cfg.IS_COVERAGE:
        coverage = Variable(torch.zeros((B, max_input_length))).to(device)

    loss = 0
    # for ei in range(input_length):
    #     encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
    #     try:
    #         encoder_outputs[ei] = encoder_output[0, 0]
    #         if cfg.LSTM:
    #             encoder_hiddens[0, ei] = encoder_hidden[0].view(-1) # [0] get h, drop c
    #         else:
    #             encoder_hiddens[0, ei] = encoder_hidden.view(-1)
    #     except:
    #         pdb.set_trace()
    sorted_enc_lens = input_mask.sum(1).sort(descending=True)[0]
    sorted_input_tensor = input_tensor[input_mask.sum(1).argsort(descending=True)]
    encoder_outputs, encoder_hidden = encoder(sorted_input_tensor, sorted_enc_lens)

    decoder_input = 20000*torch.ones(B, 1).long().to(device)

    reduce_state = ReduceState().to(device)
    decoder_hidden = reduce_state(encoder_hidden)

    use_teacher_forcing = True # False if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length):

            final_dist, decoder_hidden,  c_t_1, attn_dist, p_gen, next_coverage = decoder(encoder_outputs,
                                                                                          decoder_input, decoder_hidden,
                                                                                          c_t_1, coverage, input_idx, di,
                                                                                          input_mask)
            target = target_tensor[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            loss += -torch.log(gold_probs + cfg.EPS)

            if cfg.IS_COVERAGE:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                loss += cfg.COV_LOSS_WT * step_coverage_loss
                coverage = next_coverage

            step_mask = target_mask[:, di]
            loss = loss * step_mask

            decoder_input = target_tensor[:, di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):

            final_dist, decoder_hidden,  c_t_1, attn_dist, p_gen, next_coverage = decoder(encoder_outputs,
                                                                                          decoder_input, decoder_hidden,
                                                                                          c_t_1, coverage, input_idx, di,
                                                                                          input_mask)

            gold_probs = torch.gather(final_dist, 1, target_tensor[di].view(1,-1)).squeeze()
            loss += -torch.log(gold_probs + cfg.EPS)

            if cfg.IS_COVERAGE:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)[0]
                loss += cfg.COV_LOSS_WT * step_coverage_loss
                coverage = next_coverage

            topv, topi = final_dist.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            if decoder_input.item() == 200001:
                break

    loss = torch.mean(loss)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def trainIters(encoder, decoder, n_iters, checkpoint_dir, print_every=1000, plot_every=100, learning_rate=0.005,
               save_every=1000):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    dataloader = get_dataloader(SummarizationDataset("data/finished/test.txt", "data/word2idx.json"), batch_size=12)

    criterion = nn.NLLLoss()
    start_iter = load_model(encoder, model_dir=checkpoint_dir, appendix='Encoder', iter="l")
    if cfg.SIMPLE_ATTENTION:
        start_iter_ = load_model(decoder, model_dir=checkpoint_dir, appendix='Decoder', iter="l")
    else:
        start_iter_ = load_model(decoder, model_dir=checkpoint_dir, appendix='Decoder_full', iter="l")
    assert start_iter == start_iter_

    data_iter = iter(dataloader)

    if start_iter < n_iters:

        for i in range(start_iter, n_iters):
            try:
                batch = next(data_iter)
            except:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_tensor = batch[0].to(device)
            input_mask = batch[1].to(device)
            target_tensor = batch[2].to(device)
            target_mask = batch[3].to(device)

            if cfg.SIMPLE_ATTENTION:
                loss = train(input_tensor, target_tensor, encoder,
                             decoder, encoder_optimizer, decoder_optimizer)
            else:
                loss = train_full(input_tensor, input_mask, target_tensor, target_mask, encoder,
                             decoder, encoder_optimizer, decoder_optimizer)
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
                if cfg.SIMPLE_ATTENTION:
                    save_model(decoder, model_dir=checkpoint_dir, appendix="Decoder", iter=i + 1, save_num=3,
                           save_step=save_every)
                else:
                    save_model(decoder, model_dir=checkpoint_dir, appendix="Decoder_full", iter=i + 1, save_num=3,
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
    encoder1 = EncoderRNN(weights, cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, 1).to(device)
    if cfg.SIMPLE_ATTENTION:
        attn_decoder1 = AttnDecoderRNN(weights, 2*cfg.HIDDEN_SIZE, 200003, 1, dropout_p=0.1).to(device)
    else:
        attn_decoder1 = AttnDecoderRNN_full(weights).to(device)

    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    trainIters(encoder1, attn_decoder1, 1000000, checkpoint_dir, print_every=10, save_every=1000)