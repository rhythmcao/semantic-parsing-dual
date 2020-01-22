#coding=utf8
import torch
import torch.nn as nn
from models.model_utils import rnn_wrapper

class RNNEncoder(nn.Module):
    """
        Core encoder is a stack of N RNN layers
    """
    def __init__(self, src_emb_size, hidden_dim, num_layers, cell="lstm", bidirectional=True, dropout=0.5):
        super(RNNEncoder, self).__init__()
        self.src_emb_size = src_emb_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout if self.num_layers > 1 else 0
        self.cell = cell.upper()
        self.rnn_encoder = getattr(nn, self.cell)(self.src_emb_size, self.hidden_dim, 
                        num_layers=self.num_layers, bidirectional=self.bidirectional, 
                        batch_first=True, dropout=self.dropout)
        
    def forward(self, x, lens):
        """
            Pass the x and lens through each RNN layer.
        """
        out, hidden_states = rnn_wrapper(self.rnn_encoder, x, lens, cell=self.cell)  # bsize x srclen x dim
        return out, hidden_states