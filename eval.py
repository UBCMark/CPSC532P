import torch
import os, sys, random, pdb
import argparse

import models
from data import *
from utils import *
from pyrouge import Rouge155


system_dir = 'evaluation/sys_folder/'
model_dir = 'evaluation/model_folder/'
fname = 'summary'
system_filename_pattern = fname + '.(\d+).txt'
model_filename_pattern = fname + '.#ID#.txt'


Decoder_MAX_LENGTH = 100
r = Rouge155()

def beam_search(encoder, decoder, input_tensor, max_length=MAX_LENGTH):
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

        for di in range(Decoder_MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_hiddens)

            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if index2word[str(topi.item())] == SENTENCE_END:
                decoded_words.append(SENTENCE_END)
                break
            else:
                decoded_words.append(index2word[str(topi.item())])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def beam_decode(target_tensor, decoder_hiddens, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 5
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    input_length = input_tensor.size()[0]
    encoder_hidden = encoder.initHidden(device)
    encoder_hiddens = torch.zeros(max_length, encoder.hidden_size, device=device)

    # decoding goes sentence by sentence
    for idx in range(input_length):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch




def evaluate(encoder, decoder, checkpoint_dir, n_iters):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="Whether to evaluate model on GPU",
                        action="store_true", default=False)


    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        extractTestSum()
    if not os.path.exists(system_dir):
        os.makedirs(system_dir)

    evaluate(encoder, decoder, checkpoint_dir=checkpoint_dir, n_iters=11490)

