from pyrouge import Rouge155
import torch
import json
import os, sys
from data import cfg
from models.EncoderRNN import EncoderRNN
from models.AttnDecoderRNN import AttnDecoderRNN
from models.AttnDecoderRNN_bi import AttnDecoderRNN as AttnDecoderRNN_bi
from models.EncoderRNN_bi import EncoderRNN as EncoderRNN_bi
from data.cfg import SENTENCE_START, SENTENCE_END
from data.dataset import SummarizationDataset
from data.dataset import get_dataloader
from train import MAX_LENGTH
from utils.model_saver_iter import load_model, save_model
import pdb
import random
from test import extractTestSum

Decoder_MAX_LENGTH = 100
r = Rouge155()
system_dir = 'evaluation/sys_folder/'
model_dir = 'evaluation/model_folder/'
fname = 'summary'
system_filename_pattern = fname + '.(\d+).txt'
model_filename_pattern = fname + '.#ID#.txt'
testfile = 'data/finished/test.txt'
with open('data/idx2word.json') as json_file:
    index2word = json.load(json_file)


def evaluate(encoder, decoder, input_tensor, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden(device)

        encoder_hiddens = torch.zeros(max_length, 2*encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_hiddens[ei] += encoder_hidden[0, 0]

        decoder_input = torch.tensor([[20000]], device=device)  # SOS

        decoder_hidden = encoder_hidden.view(1, 1, -1)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        di = 0
        while di < Decoder_MAX_LENGTH:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_hiddens)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(3)
            topi = random.sample(topi.cpu().numpy().tolist()[0], 1)[0]
            if index2word[str(topi)] == SENTENCE_END:
                decoded_words.append(SENTENCE_END)
                break
            else:
                decoded_words.append(index2word[str(topi)])

            decoder_input = torch.tensor([topi], device=device).detach()
            di+=1
        return decoded_words, decoder_attentions[:di + 1]


def evaluateAll(encoder, decoder, checkpoint_dir, n_iters):
    dataloader = get_dataloader(SummarizationDataset("data/finished/test.txt", "data/word2idx.json"))
    load_model(encoder, model_dir=checkpoint_dir, appendix='Encoder', iter="l")
    load_model(decoder, model_dir=checkpoint_dir, appendix='Decoder', iter="l")

    data_iter = iter(dataloader)

    for i in range(1, n_iters):
        try:
            batch = next(data_iter)
        except:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_tensor = batch[0][0].to(device)

        output_words, _ = evaluate(encoder, decoder, input_tensor)
        output_sentence = ' '.join(output_words)
        outf = fname + '.' + str(i) + '.txt'
        fmod = open(system_dir + outf, 'w+')
        fmod.write(output_sentence)
        fmod.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python test.py <checkpoint_dir>")
        sys.exit()

    checkpoint_dir = sys.argv[1]
    weights = torch.load("data/GloVe_embeddings.pt")
    device = torch.device('cuda:2')
    bi_enable = True
    if bi_enable:
        encoder1 = EncoderRNN_bi(weights, cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, 1, dropout_p=0.1).to(device)
        attn_decoder1 = AttnDecoderRNN_bi(weights, 2 * cfg.HIDDEN_SIZE, 200003, 1, dropout_p=0.1).to(device)
    else:
        encoder1 = EncoderRNN(weights, cfg.EMBEDDING_SIZE, cfg.HIDDEN_SIZE, 2, dropout_p=0.1).to(device)
        attn_decoder1 = AttnDecoderRNN(weights, cfg.HIDDEN_SIZE, 200003, 2, dropout_p=0.1).to(device)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # extractTestSum() Only run once to extract reference summaries from test set
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        extractTestSum()
    if not os.path.exists(system_dir):
        os.makedirs(system_dir)

    r = Rouge155()
    r.system_dir = system_dir
    r.model_dir = model_dir
    r.system_filename_pattern = system_filename_pattern
    r.model_filename_pattern = model_filename_pattern

    evaluateAll(encoder1, attn_decoder1, checkpoint_dir=checkpoint_dir, n_iters=11490)
    output = r.convert_and_evaluate()
    print(output)
