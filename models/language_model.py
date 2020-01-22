#coding=utf8

''' Language Model '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import rnn_wrapper, lens2mask

class LanguageModel(nn.Module):
    """
        Container module with an encoder, a recurrent module, and a decoder.
    """
    def __init__(self, vocab_size=950, emb_size=1024, hidden_dim=256,
            num_layers=1, cell='lstm', pad_token_idxs=[], dropout=0.5,
            decoder_tied=False, init=0.2, **kargs):
        super(LanguageModel, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.cell = cell.upper() # RNN/LSTM/GRU
        self.rnn = getattr(nn, self.cell)(
            emb_size, hidden_dim, num_layers,
            batch_first=True, dropout=(dropout if num_layers > 1 else 0)
        )
        self.affine = nn.Linear(hidden_dim, emb_size)
        self.decoder = nn.Linear(emb_size, vocab_size)

        if decoder_tied:
            self.decoder.weight = self.encoder.weight # shape: vocab_size, emb_size

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_token_idxs = list(pad_token_idxs)

        if init:
            for p in self.parameters():
                p.data.uniform_(-init, init)
            for pad_token_idx in pad_token_idxs:
                self.encoder.weight.data[pad_token_idx].zero_()

    def pad_embedding_grad_zero(self):
        for pad_token_idx in self.pad_token_idxs:
            self.encoder.weight.grad[pad_token_idx].zero_()

    def forward(self, input_feats, lens):
        input_feats, lens = input_feats[:, :-1], lens - 1
        emb = self.dropout_layer(self.encoder(input_feats)) # bsize, seq_length, emb_size
        output, _ = rnn_wrapper(self.rnn, emb, lens, self.cell)
        decoded = self.decoder(self.affine(self.dropout_layer(output)))
        scores = F.log_softmax(decoded, dim=-1)
        return scores

    def sent_logprobability(self, input_feats, lens):
        '''
            Given sentences, calculate its length-normalized log-probability
            Sequence must contain <s> and </s> symbol
            lens: length tensor
        '''
        lens = lens - 1
        input_feats, output_feats = input_feats[:, :-1], input_feats[:, 1:]
        emb = self.dropout_layer(self.encoder(input_feats)) # bsize, seq_len, emb_size
        output, _ = rnn_wrapper(self.rnn, emb, lens, self.cell)
        decoded = self.decoder(self.affine(self.dropout_layer(output)))
        scores = F.log_softmax(decoded, dim=-1)
        log_prob = torch.gather(scores, 2, output_feats.unsqueeze(-1)).contiguous().view(output.size(0), output.size(1))
        sent_log_prob = torch.sum(log_prob * lens2mask(lens).float(), dim=-1)
        return sent_log_prob / lens.float()

    def load_model(self, load_dir):
        self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))
