#coding=utf8
"""
    Word2vec utilities: load pre-trained embeddings: Glove6B embeddings
"""
import torch
import numpy as np
from collections import defaultdict
from utils.constants import BOS, EOS, PAD, UNK, VECTORCACHE

def read_pretrained_vectors(filename, vocab, device):
    word2vec, mapping = {}, {}
    mapping['bos'], mapping['eos'], mapping['padding'], mapping['unknown'] = BOS, EOS, PAD, UNK
    with open(filename, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line == '':
                continue
            word = line[:line.index(' ')]
            word = mapping[word] if word in mapping else word
            if word in vocab:
                values = line[line.index(' ') + 1:]
                word2vec[word] = torch.tensor(np.fromstring(values, sep=' ', dtype=np.float), device=device)
    return word2vec

def load_embeddings(module, word2id, device=None):
    emb_dim = module.weight.data.size(-1)
    if emb_dim not in [50, 100, 200, 300]:
        print('Not use pretrained glove6B embeddings ...')
        return 0.0
    word2vec_file = VECTORCACHE(emb_dim)
    pretrained_vectors = read_pretrained_vectors(word2vec_file, word2id, device)
    for word in pretrained_vectors:
        module.weight.data[word2id[word]] = pretrained_vectors[word]
    return len(pretrained_vectors)/float(len(word2id))
