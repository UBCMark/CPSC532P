import torch
import os, sys, random, pdb
import argparse

import models
from data import *
from utils import *
from pyrouge import Rouge155
from queue import PriorityQueue
from models.AttnDecoderRNN_bi import AttnDecoderRNN, AttnDecoderRNN_full, Attention, ReduceState
from models.EncoderRNN_bi import EncoderRNN

Decoder_MAX_LENGTH = 100
system_dir = 'evaluation/sys_folder/'
model_dir = 'evaluation/model_folder/'
fname = 'summary'
system_filename_pattern = fname + '.(\d+).txt'
model_filename_pattern = fname + '.#ID#.txt'


Decoder_MAX_LENGTH = 100
r = Rouge155()


def get_top_k(decoder_output, k=5):

    tokens = [[]] * k
    beam = [[]] * k
    score = [0] * k

    topv, topi = torch.topk(decoder_output, beam_width)

        for k in range(beam_width):

            token = topi[k].squeeze().detach()
            tokens[k].append(token)
            word = index2word[str(topi[k].item())]
            beam[k].append(word)
            score[k] += abs(topv[k].item())

    return tokens, beam, score


def beam_search(encoder, decoder, input_tensor, max_length=MAX_LENGTH, beam_width=5):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden(device)

        encoder_hiddens = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_hiddens[ei] += encoder_hidden[0, 0]

        decoder_input = torch.tensor([[20000]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)


        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_hiddens)
        decoder_attentions[di] = decoder_attention.data

        tokens, beam, score = get_top_k(decoder_output)
        decoder_hiddens = beam_width * [decoder_hidden]

        while True:

            if all([b[-1] == cfg.SENTENCE_END for b in beam]):
                break

            candidates = []

            for i, token in enumerate(tokens):

                if beam[i][-1] == cfg.SENTENCE_END or len(beam[i]) >= Decoder_MAX_LENGTH:
                    candidates = append((token, beam[i], score[i], len(beam[i]), None))

                decoder_output, new_decoder_hidden, decoder_attention = decoder(
                token, decoder_hiddens[i], encoder_hiddens)

                cur_tokens, cur_beam, cur_score = get_top_k(decoder_output, k=5)

                for j in range(len(cur_tokens)):
                    candidates.append((cur_tokens[j],
                                       beam[i] + cur_beam[j],
                                       score[i] + cur_score[j],
                                       len(beam[i]) + 1,
                                       new_decoder_hidden))

            candidates.sort(key=lambda candidate: (candidate[2] / candidate[3]))

            candidates = candidates[:5]

            tokens = [candidate[0] for candidate in candidates]
            beam = [candidate[1] for candidate in candidates]
            score = [candidate[2] for candidate in candidates]
            decoder_hiddens = [candidates[3] for candidate in candidates]


        return beam[0], decoder_attentions[:di + 1]


def evaluate(encoder, decoder, checkpoint_dir, n_iters):
    dataloader = get_dataloader(SummarizationDataset("data/finished/test.txt", "data/word2idx.json"))


    encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'encoder_4.pth')))
    decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'decoder_4.pth')))
    # load_model(encoder, model_dir=checkpoint_dir, appendix='Encoder', iter="l")
    # load_model(decoder, model_dir=checkpoint_dir, appendix='Decoder', iter="l")

    data_iter = iter(dataloader)

    for i in range(1, n_iters):
        try:
            batch = next(data_iter)
        except:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_tensor = batch[0][0].to(device)

        output_words, _ = beam_search(encoder, decoder, input_tensor)
        output_sentence = ' '.join(output_words)
        outf = fname + '.' + str(i) + '.txt'
        fmod = open(system_dir + outf, 'w+')
        fmod.write(output_sentence)
        fmod.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="Whether to evaluate model on GPU",
                        action="store_true", default=False)

    weights = torch.load("data/GloVe_embeddings.pt")
    device = torch.device('cuda:0')
    bi_enable = True
    if bi_enable:
        encoder1 = EncoderRNN(weights, cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, 1).to(device)
        attn_decoder1 = EncoderRNN(weights, 2 * cfg.HIDDEN_SIZE, 200003, 1).to(device)
    else:
        encoder1 = EncoderRNN(weights, cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, 2).to(device)
        attn_decoder1 = AttnDecoderRNN(weights, cfg.HIDDEN_SIZE, 200003, 2).to(device)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        extractTestSum()
    if not os.path.exists(system_dir):
        os.makedirs(system_dir)


    evaluate(encoder, decoder, checkpoint_dir=checkpoint_dir, n_iters=11490)

