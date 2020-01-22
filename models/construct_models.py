#coding=utf8
import torch
import torch.nn as nn
from utils.constants import *
from models.embedding.embedding_rnn import RNNEmbeddings
from models.encoder.encoder_rnn import RNNEncoder
from models.enc2dec.state_transition import StateTransition
from models.attention.attention_rnn import Attention
from models.decoder.decoder_rnn import RNNDecoder
from models.decoder.decoder_rnn_pointer import RNNDecoderPointer
from models.generator.generator_naive import Generator
from models.generator.generator_pointer import GeneratorPointer
from models.model_attn import AttnModel
from models.model_attnptr import AttnPtrModel

def construct_model(*args, **kargs):
    copy = kargs.pop('copy', True)
    if copy:
        return construct_attnptr(*args, **kargs)
    else:
        return construct_attn(*args, **kargs)

def construct_attn(
    src_vocab=None, tgt_vocab=None, src_unk_idx=1, tgt_unk_idx=1, pad_src_idxs=[0], pad_tgt_idxs=[0],
    src_emb_size=100, tgt_emb_size=100, hidden_dim=200, num_layers=1, bidirectional=True,
    cell='lstm', dropout=0.5, init=None, **kargs
):
    """
        Construct Seq2Seq model with attention mechanism
    """
    num_directions = 2 if bidirectional else 1
    enc2dec_model = StateTransition(num_layers, cell=cell, bidirectional=bidirectional, hidden_dim=hidden_dim)
    attn_model = Attention(hidden_dim * num_directions, hidden_dim)
    src_embeddings = RNNEmbeddings(src_emb_size, src_vocab, src_unk_idx, pad_token_idxs=pad_src_idxs, dropout=dropout)
    encoder = RNNEncoder(src_emb_size, hidden_dim, num_layers, cell=cell, bidirectional=bidirectional, dropout=dropout)
    tgt_embeddings = RNNEmbeddings(tgt_emb_size, tgt_vocab, tgt_unk_idx, pad_token_idxs=pad_tgt_idxs, dropout=dropout)
    decoder = RNNDecoder(tgt_emb_size, hidden_dim, num_layers, attn=attn_model, cell=cell, dropout=dropout)
    generator_model = Generator(tgt_emb_size, tgt_vocab, dropout=dropout)
    model = AttnModel(src_embeddings, encoder, tgt_embeddings, decoder, enc2dec_model, generator_model)

    if init:
        for p in model.parameters():
            p.data.uniform_(-init, init)
        for pad_token_idx in pad_src_idxs:
            model.src_embed.embed.weight.data[pad_token_idx].zero_()
        for pad_token_idx in pad_tgt_idxs:
            model.tgt_embed.embed.weight.data[pad_token_idx].zero_()
    return model

def construct_attnptr(
    src_vocab=None, tgt_vocab=None, src_unk_idx=1, tgt_unk_idx=1, pad_src_idxs=[0], pad_tgt_idxs=[0],
    src_emb_size=100, tgt_emb_size=100, hidden_dim=200, bidirectional=True, num_layers=1,
    cell='lstm', dropout=0.5, init=None, **kargs
):
    """
        Construct Seq2Seq model with attention mechanism and pointer network
    """
    num_directions = 2 if bidirectional else 1
    enc2dec_model = StateTransition(num_layers, cell=cell, bidirectional=bidirectional, hidden_dim=hidden_dim)
    attn_model = Attention(hidden_dim * num_directions, hidden_dim)
    src_embeddings = RNNEmbeddings(src_emb_size, src_vocab, src_unk_idx, pad_token_idxs=pad_src_idxs, dropout=dropout)
    encoder = RNNEncoder(src_emb_size, hidden_dim, num_layers, cell=cell, bidirectional=bidirectional, dropout=dropout)
    tgt_embeddings = RNNEmbeddings(tgt_emb_size, tgt_vocab, tgt_unk_idx, pad_token_idxs=pad_tgt_idxs, dropout=dropout)
    decoder = RNNDecoderPointer(tgt_emb_size, hidden_dim, num_layers, attn=attn_model, cell=cell, dropout=dropout)
    generator_model = GeneratorPointer(tgt_emb_size, tgt_vocab, dropout=dropout)
    model = AttnPtrModel(src_embeddings, encoder, tgt_embeddings, decoder, enc2dec_model, generator_model)

    if init:
        for p in model.parameters():
            p.data.uniform_(-init, init)
        for pad_token_idx in pad_src_idxs:
            model.src_embed.embed.weight.data[pad_token_idx].zero_()
        for pad_token_idx in pad_tgt_idxs:
            model.tgt_embed.embed.weight.data[pad_token_idx].zero_()
    return model
