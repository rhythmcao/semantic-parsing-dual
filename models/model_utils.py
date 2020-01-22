#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

def tile(x, count, dim=0):
    """
        Tiles x on dimension dim count times.
        E.g. [1, 2, 3], count=2 ==> [1, 1, 2, 2, 3, 3]
            [[1, 2], [3, 4]], count=3, dim=1 ==> [[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]
        Different from torch.repeat
    """
    if x is None:
        return x
    elif type(x) in [list, tuple]:
        return type(x)([tile(each, count, dim) for each in x])
    else:
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.contiguous().view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x

def lens2mask(lens):
    bsize = lens.numel()
    max_len = lens.max()
    masks = torch.arange(0, max_len).type_as(lens).to(lens.device).repeat(bsize, 1).lt(lens.unsqueeze(1))
    masks.requires_grad = False
    return masks

def rnn_wrapper(encoder, inputs, lens, cell='lstm'):
    """
        @args:
            encoder(nn.Module): rnn series bidirectional encoder, batch_first=True
            inputs(torch.FloatTensor): rnn inputs, bsize x max_seq_len x in_dim
            lens(torch.LongTensor): seq len for each sample, bsize
        @return:
            out(torch.FloatTensor): output of encoder, bsize x max_seq_len x hidden_dim*2
            hidden_states(tuple or torch.FloatTensor): final hidden states, num_layers*2 x bsize x hidden_dim
    """
    # rerank according to lens and temporarily remove empty inputs
    sorted_lens, sort_key = torch.sort(lens, descending=True)
    nonzero_index = torch.sum(sorted_lens > 0).item()
    sorted_inputs = torch.index_select(inputs, dim=0, index=sort_key[:nonzero_index])
    # forward non empty inputs    
    packed_inputs = rnn_utils.pack_padded_sequence(sorted_inputs, sorted_lens[:nonzero_index].tolist(), batch_first=True)
    packed_out, h = encoder(packed_inputs)  # bsize x srclen x dim
    out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
    if cell.upper() == 'LSTM':
        h, c = h
    # pad zeros due to empty inputs
    pad_zeros = torch.zeros(sorted_lens.size(0) - out.size(0), out.size(1), out.size(2)).type_as(out).to(out.device)
    sorted_out = torch.cat([out, pad_zeros], dim=0)
    pad_hiddens = torch.zeros(h.size(0), sorted_lens.size(0) - h.size(1), h.size(2)).type_as(h).to(h.device)
    sorted_hiddens = torch.cat([h, pad_hiddens], dim=1)
    if cell.upper() == 'LSTM': 
        pad_cells = torch.zeros(c.size(0), sorted_lens.size(0) - c.size(1), c.size(2)).type_as(c).to(c.device)
        sorted_cells = torch.cat([c, pad_cells], dim=1)
    # rerank according to sort_key
    shape = list(sorted_out.size())
    out = torch.zeros_like(sorted_out).type_as(sorted_out).to(sorted_out.device).scatter_(0, sort_key.unsqueeze(-1).unsqueeze(-1).expand(*shape), sorted_out)
    shape = list(sorted_hiddens.size())
    hiddens = torch.zeros_like(sorted_hiddens).type_as(sorted_hiddens).to(sorted_hiddens.device).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).expand(*shape), sorted_hiddens)
    if cell.upper() == 'LSTM':
        cells = torch.zeros_like(sorted_cells).type_as(sorted_cells).to(sorted_cells.device).scatter_(1, sort_key.unsqueeze(0).unsqueeze(-1).expand(*shape), sorted_cells)
        return out, (hiddens.contiguous(), cells.contiguous())
    return out, hiddens.contiguous()
