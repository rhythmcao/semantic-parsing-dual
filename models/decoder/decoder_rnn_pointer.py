#coding=utf8
import torch
import torch.nn as nn

class RNNDecoderPointer(nn.Module):
    """
        Generic unidirectional RNN layers containing StateTransition and Attention modules.
    """
    def __init__(self, tgt_emb_size, hidden_dim, num_layers, attn, cell="lstm", dropout=0.5):
        super(RNNDecoderPointer, self).__init__()
        self.tgt_emb_size = tgt_emb_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout if self.num_layers > 1 else 0
        self.cell = cell.upper()
        self.rnn_decoder = getattr(nn, self.cell)(self.tgt_emb_size, self.hidden_dim,
            num_layers=self.num_layers, bidirectional=False, batch_first=True, dropout=self.dropout)
        self.attn = attn
        self.gate = nn.Linear(self.hidden_dim + self.attn.enc_dim + self.tgt_emb_size, 1)
        self.affine = nn.Linear(self.hidden_dim + self.attn.enc_dim, self.tgt_emb_size)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, hidden_states, memory, src_mask, copy_tokens=None):
        """
            x: decoder input embeddings, bsize x tgt_len x emb_size
            hidden_states: previous decoder state
            memory: memory and hidden_states
            src_mask: mask on src input, bsize x src_lens
            copy_tokens: bsize x src_lens x vocab_size
            @return:
                feats: bsize x tgt_lens x (dec_dim + enc_dim)
                copy_distribution: bsize x tgt_lens x (vocab_size + MAX_OOV_NUM)
                gate_scores: bsize x tgt_lens x 1
        """
        out, hidden_states = self.rnn_decoder(x, hidden_states)
        context, pointer = [], []
        for i in range(out.size(1)):
            tmp_context, tmp_ptr = self.attn(memory, out[:, i, :], src_mask)
            context.append(tmp_context)
            pointer.append(tmp_ptr.unsqueeze(dim=1))
        context, pointer = torch.cat(context, dim=1), torch.cat(pointer, dim=1)
        feats = self.dropout_layer(torch.cat([out, context], dim=-1))
        gate_scores = torch.sigmoid(self.gate(torch.cat([feats, x], dim=-1)))
        feats = self.affine(feats)
        copy_distribution = torch.bmm(pointer, copy_tokens)
        return feats, hidden_states, copy_distribution, gate_scores
