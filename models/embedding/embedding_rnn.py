#coding=utf8
import torch
import torch.nn as nn

class RNNEmbeddings(nn.Module):
    def __init__(self, emb_size, vocab, unk_idx=1, pad_token_idxs=[0], dropout=0.5):
        super(RNNEmbeddings, self).__init__()
        self.embed = nn.Embedding(vocab, emb_size)
        self.vocab = vocab
        self.emb_size = emb_size
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pad_token_idxs = pad_token_idxs
        self.unk_idx = unk_idx

    def forward(self, x):
        token_mask = x >= self.vocab
        if token_mask.any():
            x = x.masked_fill_(token_mask, self.unk_idx)
        return self.dropout_layer(self.embed(x))
    
    def pad_embedding_grad_zero(self):
        for pad_token_idx in self.pad_token_idxs:
            self.embed.weight.grad[pad_token_idx].zero_()