#coding=utf8
import torch
import torch.nn as nn

class RNNDecoder(nn.Module):
    """
        Generic unidirectional RNN layers containing Attention modules.
    """
    def __init__(self, tgt_emb_size, hidden_dim, num_layers, attn, cell="lstm", dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.tgt_emb_size = tgt_emb_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout if self.num_layers > 1 else 0
        self.cell = cell.upper()
        self.rnn_decoder = getattr(nn, self.cell)(self.tgt_emb_size, self.hidden_dim,
            num_layers=self.num_layers, bidirectional=False, batch_first=True, dropout=self.dropout)
        self.attn = attn
        self.affine = nn.Linear(self.hidden_dim + self.attn.enc_dim, self.tgt_emb_size)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, hidden_states, memory, src_mask, copy_tokens=None):
        """
            x: decoder input embeddings, bsize x tgt_len x emb_size
            hidden_states: previous decoder state
            memory: encoder output, bsize x src_len x hidden_dim*2
            src_mask: bsize x src_lens
            copy_tokens: to be compatible with pointer network
        """
        out, hidden_states = self.rnn_decoder(x, hidden_states)
        context = []
        for i in range(out.size(1)):
            tmp_context, _ = self.attn(memory, out[:, i, :], src_mask)
            context.append(tmp_context)
        context = torch.cat(context, dim=1)
        feats = torch.cat([out, context], dim=-1)
        feats = self.affine(self.dropout_layer(feats))
        return feats, hidden_states
