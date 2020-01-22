#coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    """
        A standard Encoder-Decoder architecture.
    """
    def __init__(self, src_embed, encoder, tgt_embed, decoder, enc2dec, generator):
        """
            All the arguments are of type nn.Module
        """
        super(EncoderDecoder, self).__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.enc2dec = enc2dec
        self.tgt_embed = tgt_embed
        self.decoder = decoder
        self.generator = generator
    
    def forward(self, *args, **kargs):
        raise NotImplementedError

    def decode_batch(self, *args, **kargs):
        raise NotImplementedError

    def decode_greed(self, *args, **kargs):
        raise NotImplementedError
    
    def decode_beam_search(self, *args, **kargs):
        raise NotImplementedError

    def pad_embedding_grad_zero(self):
        self.src_embed.pad_embedding_grad_zero()
        self.tgt_embed.pad_embedding_grad_zero()

    def load_model(self, load_dir):
        self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))