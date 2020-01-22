#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
        Define standard linear + softmax generation step.
    """
    def __init__(self, feats, vocab, dropout=0.5):
        super(Generator, self).__init__()
        self.proj = nn.Linear(feats, vocab)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        return F.log_softmax(self.proj(self.dropout_layer(x)), dim=-1)