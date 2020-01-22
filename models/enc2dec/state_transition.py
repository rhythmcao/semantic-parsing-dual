#coding=utf8
import torch
import torch.nn as nn

class StateTransition(nn.Module):

    METHODS = ['affine', 'reverse', 'tanh(affine)', 'empty']

    def __init__(self, num_layers, cell='lstm', bidirectional=True, hidden_dim=None, method='empty'):
        """
            Transform encoder final hidden states to decoder initial hidden states
        """
        super(StateTransition, self).__init__()
        self.cell = cell.upper()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        assert method in StateTransition.METHODS
        self.method = method
        if 'affine' in self.method:
            assert hidden_dim
            self.h_affine = nn.Linear(hidden_dim * self.num_directions, hidden_dim)
            if self.cell == 'LSTM':
                self.c_affine = nn.Linear(hidden_dim * self.num_directions, hidden_dim)

    def forward(self, hidden_states):
        if self.method == 'empty':
            if 'LSTM' in self.cell:
                enc_h, enc_c = hidden_states
                dec_h = enc_h.new_zeros(self.num_layers, enc_h.size(1), enc_h.size(2))
                dec_c = enc_c.new_zeros(self.num_layers, enc_c.size(1), enc_c.size(2))
                hidden_states = (dec_h, dec_c)
            else:
                enc_h = hidden_states
                dec_h = enc_h.new_zeros(self.num_layers, enc_h.size(1), enc_h.size(2))
                hidden_states = dec_h
        elif self.method == 'reverse':
            if self.num_directions == 2:
                index_slices = [2 * i + 1 for i in range(self.num_layers)]  # from reversed path
                index_slices = torch.tensor(index_slices, dtype=torch.long, device=hidden_states[0].device)
                if self.cell == 'LSTM':
                    enc_h, enc_c = hidden_states
                    dec_h = torch.index_select(enc_h, 0, index_slices)
                    dec_c = torch.index_select(enc_c, 0, index_slices)
                    hidden_states = (dec_h.contiguous(), dec_c.contiguous())
                else:
                    enc_h = hidden_states
                    dec_h = torch.index_select(enc_h, 0, index_slices)
                    hidden_states = dec_h.contiguous()
            else:
                pass # do nothing, pass states directly
        else:
            if self.cell == 'LSTM':
                enc_h, enc_c = hidden_states
                batches = enc_h.size(1)
                dec_h = self.h_affine(enc_h.transpose(0, 1).contiguous().view(batches * self.num_layers, -1))
                dec_c = self.c_affine(enc_c.transpose(0, 1).contiguous().view(batches * self.num_layers, -1))
                if "tanh" in self.method:
                    dec_h, dec_c = torch.tanh(dec_h), torch.tanh(dec_c)
                dec_h = dec_h.contiguous().view(batches, self.num_layers, -1).transpose(0, 1).contiguous()
                dec_c = dec_c.contiguous().view(batches, self.num_layers, -1).transpose(0, 1).contiguous()
                hidden_states = (dec_h, dec_c)
            else:
                enc_h, batches = hidden_states, hidden_states.size(1)
                dec_h = self.h_affine(enc_h.transpose(0, 1).contiguous().view(batches * self.num_layers, -1))
                if "tanh" in self.method:
                    dec_h = torch.tanh(dec_h)
                dec_h = dec_h.contiguous().view(batches, self.num_layers, -1).transpose(0, 1).contiguous()
                hidden_states = dec_h
        return hidden_states
