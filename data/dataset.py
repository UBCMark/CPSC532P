import torch
from torch.utils.data import Dataset, DataLoader
import json, os

if __name__ == "__main__":
    import cfg
else:
    from . import cfg

import pdb


class SummarizationDataset(Dataset):
    '''
    Dataset for loading articles and target summary
    '''

    def __init__(self, txt_path, map_path):
        """
        txt_path: path to txt file
        emb_path: path to word2vec embeddings
        map_path: path to vocab to embeddings mapping file
        """

        if not os.path.isfile(txt_path):
            raise Exception("Data file not found at location \"{}\".".format(txt_path))
        # if not os.path.isfile(emb_path):
        #     raise Exception("Embedding file not found at location \"{}\".".format(emb_path))
        if not os.path.isfile(map_path):
            raise Exception("JSON mapping file not found at location \"{}\".".format(map_path))

        print("Loading txt file into memory...")
        self.data = []
        self.target = []

        with open(txt_path, 'r') as reader:
            for i, line in enumerate(reader):
                tokens = line.split()

                if i % 2 == 0:
                    self.data.append(tokens)
                elif i % 2 == 1:
                    self.target.append(tokens)

        over_limit = set([i for i in range(len(self.data)) if len(self.data[i]) > cfg.INPUT_MAX] + \
                         [i for i in range(len(self.target)) if len(self.target[i]) > cfg.OUTPUT_MAX])
        self.data = [x for i, x in enumerate(self.data) if i not in over_limit]
        self.target = [x for i, x in enumerate(self.target) if i not in over_limit]

        print("Finished loading txt file.")

        if not len(self.data) == len(self.target):
            raise Exception("Number of articles do not match the number of summaries.")

        if not len(self.data) == len(self.target):
            raise Exception("JSON mapping file not found at location \"{}\".".format(map_path))

        # self.emb = torch.load(emb_path)

        # if not self.emb.size() == torch.Size([cfg.VOCAB_SIZE + 2, cfg.EMBEDDING_SIZE]):
        #     raise Exception("Embeddings do not have the correct shape.")

        with open(map_path, 'r') as json_file:
            self.map = json.load(json_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx]
        target = self.target[idx]
        data = [cfg.SENTENCE_START] + data + [cfg.SENTENCE_END]
        target = target + [cfg.SENTENCE_END]

        # input_emb = torch.zeros((len(data), cfg.EMBEDDING_SIZE))

        # word_indices = []
        # for token in data:
        #     if token in self.map.keys():
        #         word_indices.append(self.map[token])
        #     else:
        #         word_indices.append(-1)

        # for i, word_idx in enumerate(word_indices):
        #     if word_idx != -1:
        #         input_emb[i] = self.emb[word_idx]

        input_idx = torch.zeros(500, dtype=torch.long)
        input_mask = torch.zeros(500)
        input_mask[:len(data)] = 1
        for i, token in enumerate(data):
            try:
                input_idx[i] = self.map[token]

            # Out of vocab
            except:
                input_idx[i] = self.map[cfg.UNKNOWN]
        

        target_idx = torch.zeros(100, dtype=torch.long)
        target_mask = torch.zeros(100)
        target_mask[:len(target)] = 1
        for i, token in enumerate(target):
            try:
                target_idx[i] = self.map[token]

            # Out of vocab
            except:
                target_idx[i] = self.map[cfg.UNKNOWN]

        return input_idx, input_mask, target_idx, target_mask


def output2tokens(index, idx2word):
    """
    index: a (N x V) Tensor where N is the length of the sentence and V the length of vocab + 2
    idx2word: dict that maps embedding index to tokens
    """
    tokens = []

    for i in range(index.size()[0]):
        idx = index[i].item()
        tokens.append(idx2word[str(idx)])

    return tokens


def get_dataloader(dataset, batch_size=8):
    n_workers = os.cpu_count()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)
    return dataloader


# For testing
if __name__ == "__main__":

    with open('idx2word.json', 'r') as json_file:
        idx2word = json.load(json_file)

    dataset = SummarizationDataset(os.path.join('finished', 'val.txt'), 'word2idx.json')

    # for i in range(len(dataset)):
    #     data, target = dataset[i]
    #     pdb.set_trace()
    #     tokens = output2tokens(data, idx2word)

    dataloader = get_dataloader(dataset)

    for i, (data, data_mask, target, target_mask) in enumerate(dataloader):
        input_len = data_mask.sum(-1)
        target_len = target_mask.sum(-1)

        pdb.set_trace()
        print(data.size(), data_mask.size())