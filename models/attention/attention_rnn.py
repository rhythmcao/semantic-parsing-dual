#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    METHODS = ['general', 'feedforward']

    def __init__(self, enc_dim, dec_dim, method='feedforward'):

        super(Attention, self).__init__()
        self.enc_dim, self.dec_dim = enc_dim, dec_dim
        assert method in Attention.METHODS
        self.method = method
        if self.method == 'general':
            self.Wa = nn.Linear(self.enc_dim, self.dec_dim, bias=False)
        else:
            self.Wa = nn.Linear(self.enc_dim + self.dec_dim, self.dec_dim, bias=False)
            self.Va = nn.Linear(self.dec_dim, 1, bias=False)

    def forward(self, hiddens, decoder_state, masks):
        '''
            hiddens : bsize x src_lens x enc_dim
            decoder_state : bsize x dec_dim
            masks : bsize x src_lens, ByteTensor
            @return: 
                context : bsize x 1 x enc_dim
                a : normalized coefficient, bsize x src_lens
        '''
        if self.method == 'general':
            m = self.Wa(hiddens) # bsize x src_len x dec_dim
            e = torch.bmm(m, decoder_state.unsqueeze(-1)).squeeze(dim=-1) # bsize x src_len
        else:
            d = decoder_state.unsqueeze(dim=1).repeat(1, hiddens.size(1), 1)
            e = self.Wa(torch.cat([d, hiddens], dim=-1))
            e = self.Va(torch.tanh(e)).squeeze(dim=-1)
        e.masked_fill_(masks == 0, -float('inf'))
        a = F.softmax(e, dim=1) 
        context = torch.bmm(a.unsqueeze(1), hiddens)
        return context, a