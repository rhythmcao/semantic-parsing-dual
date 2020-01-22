#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import MAX_OOV_NUM

class GeneratorPointer(nn.Module):
    """
        Define standard linear + softmax generation step plus pointer copy step.
    """
    def __init__(self, feats, vocab, dropout=0.5):
        super(GeneratorPointer, self).__init__()
        self.proj = nn.Linear(feats, vocab)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, copy_distribution, gate_scores):
        """
            x: bsize x tgt_lens x (dec_dim + enc_dim)
            copy_distribution: bsize x tgt_lens x (vocab_size + MAX_OOV_NUM)
            gate_scores: bsize x tgt_lens x 1
        """
        out = F.softmax(self.proj(self.dropout_layer(x)), dim=-1)
        extra_zeros = torch.zeros(x.size(0), x.size(1), MAX_OOV_NUM, dtype=torch.float, device=x.device)
        generate_distribution = torch.cat([out, extra_zeros], dim=-1)
        final_scores = torch.log(gate_scores * generate_distribution + (1 - gate_scores) * copy_distribution + 1e-20)
        return final_scores