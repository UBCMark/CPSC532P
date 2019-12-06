import torch
import os, sys, random, pdb
import argparse
import json

import models
from data import *
from utils import *
from pyrouge import Rouge155
from queue import PriorityQueue
from models.AttnDecoderRNN_bi import AttnDecoderRNN, AttnDecoderRNN_full, Attention, ReduceState
from models.EncoderRNN_bi import EncoderRNN_batch as EncoderRNN
from train import MAX_LENGTH as MAX_LENGTH
from data import cfg
from data.dataset import get_dataloader
from data.dataset import SummarizationDataset
from utils.model_saver_iter import load_model, save_model
from torch.autograd import Variable

with open('data/idx2word.json') as json_file:
    index2word = json.load(json_file)


system_dir = 'evaluation/sys_folder/'
model_dir = 'evaluation/model_folder/'
fname = 'summary'
system_filename_pattern = fname + '.(\d+).txt'
model_filename_pattern = fname + '.#ID#.txt'


Decoder_MAX_LENGTH = 100
r = Rouge155()


def get_top_k(decoder_output, k=cfg.BEAM_WIDTH):

    tokens = []
    beam = [[]*k for _ in range(k)]
    score = [0] * k

    beam_width = k
    topv, topi = torch.topk(decoder_output[0], beam_width*2)
    topv, topi= topv[torch.randperm(beam_width*2)[:beam_width]], topi[torch.randperm(beam_width*2)[:beam_width]]

    for k in range(int(beam_width)):
        token = topi[k].squeeze().detach()
        tokens.append(token)
        word = index2word[str(topi[k].item())]
        beam[k].append(word)
        score[k] += abs(topv[k].item())

    return tokens, beam, score


def beam_search(encoder, decoder, input_tensor, input_mask, max_length=MAX_LENGTH, beam_width=cfg.BEAM_WIDTH):
    with torch.no_grad():
        enc_lens = input_mask.sum().int()

        input_mask = input_mask[:enc_lens[0]]
        input_tensor = input_tensor[:enc_lens[0]]

        encoder_outputs, encoder_hidden = encoder(input_tensor.view(1,-1), enc_lens.view(1))

        decoder_input = torch.tensor([[20000]], device=device)  # SOS

        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)


        input_idx = input_tensor.view(1,-1)
        reduce_state = ReduceState().to(device)
        decoder_hidden = reduce_state(encoder_hidden)

        c_t_1 = Variable(torch.zeros((1, 2 * cfg.HIDDEN_SIZE))).to(device)

        coverage = None
        if cfg.IS_COVERAGE:
            coverage = Variable(torch.zeros(enc_lens[0])).to(device)

        final_dist, decoder_hidden,  c_t_1, attn_dist, p_gen, next_coverage = decoder(encoder_outputs,
                                                                                      decoder_input, decoder_hidden,
                                                                                      c_t_1, coverage, input_idx, 0,
                                                                                      input_mask)
        if cfg.IS_COVERAGE:
            coverage = next_coverage

        tokens, beam, score = get_top_k(final_dist)
        decoder_hiddens = beam_width * [decoder_hidden]
        while True:

            if all([b[-1] == cfg.SENTENCE_END for b in beam]):
                break

            candidates = []

            for i, token in enumerate(tokens):
                if beam[i][-1] == cfg.SENTENCE_END:
                    candidates.append((token, beam[i], score[i], len(beam[i]), None))
                    continue
                if len(beam[i]) >= Decoder_MAX_LENGTH:
                    candidates.append((token, beam[i] + [cfg.SENTENCE_END], score[i], len(beam[i]), None))
                    continue

                final_dist, new_decoder_hidden, c_t_1, attn_dist, p_gen, next_coverage = decoder(encoder_outputs,
                                                                             token,
                                                                             decoder_hiddens[i],
                                                                             c_t_1, coverage,
                                                                             input_idx, 1, input_mask)
                if cfg.IS_COVERAGE:
                    coverage = next_coverage

                cur_tokens, cur_beam, cur_score = get_top_k(final_dist, k=cfg.BEAM_WIDTH)

                for j in range(len(cur_tokens)):
                    candidates.append((cur_tokens[j],
                                       beam[i] + cur_beam[j],
                                       score[i] + cur_score[j],
                                       len(beam[i]) + 1,
                                       new_decoder_hidden))

            candidates.sort(key=lambda candidate: (candidate[2] / candidate[3]))

            candidates = candidates[:cfg.BEAM_WIDTH]

            tokens = [candidate[0] for candidate in candidates]
            beam = [candidate[1] for candidate in candidates]
            score = [candidate[2] for candidate in candidates]
            decoder_hiddens = [candidate[4] for candidate in candidates]


        return beam[0] #, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, checkpoint_dir, n_iters):
    dataloader = get_dataloader(SummarizationDataset("data/finished/test.txt", "data/word2idx.json"), batch_size=1)


    #encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'encoder_4.pth')))
    #decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'decoder_4.pth')))
    load_model(encoder, model_dir=checkpoint_dir, appendix='Encoder', iter="l")
    load_model(decoder, model_dir=checkpoint_dir, appendix='Decoder_full', iter="l")

    data_iter = iter(dataloader)

    for i in range(1, n_iters):
        try:
            batch = next(data_iter)
        except:
            print("AAAA")
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_tensor = batch[0][0].to(device)
        input_mask = batch[1][0].to(device)
        output_words = beam_search(encoder, decoder, input_tensor, input_mask)
        output_sentence = ' '.join(output_words)
        outf = fname + '.' + str(i) + '.txt'
        fmod = open(system_dir + outf, 'w+')
        fmod.write(output_sentence)
        fmod.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="Whether to evaluate model on GPU",
                        action="store_true", default=False)
    if len(sys.argv) != 2:
        print("USAGE: python train.py <checkpoint_dir>")
        sys.exit()

    checkpoint_dir = sys.argv[1]
    weights = torch.load("data/GloVe_embeddings.pt")
    device = torch.device('cuda:1')
    bi_enable = True
    if bi_enable:
        encoder1 = EncoderRNN(weights, cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, 1).to(device)
        if cfg.SIMPLE_ATTENTION:
            attn_decoder1 = AttnDecoderRNN(weights, 2 * cfg.HIDDEN_SIZE, 200003, 1).to(device)
        else:
            attn_decoder1 = AttnDecoderRNN_full(weights).to(device)
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

    evaluate(encoder1, attn_decoder1, checkpoint_dir=checkpoint_dir, n_iters=11490)