#coding=utf8
'''
    Set loss function, allow different confidence for different training samples
'''
import torch
import torch.nn as nn

def set_loss_function(reduction='sum', ignore_index=-100):
    loss_function = MyNLLLoss(reduction=reduction, ignore_index=ignore_index)
    return loss_function

class MyNLLLoss(nn.Module):

    def __init__(self, *args, **kargs):
        super(MyNLLLoss, self).__init__()
        self.real_reduction = kargs.pop('reduction', 'sum')
        kargs['reduction'] = 'none'
        self.loss_function = nn.NLLLoss(*args, **kargs)

    def forward(self, inputs, targets, lens=None, conf=None):
        if conf is None:
            conf = torch.ones(inputs.size(0), dtype=torch.float).to(inputs.device)
        bsize, seq_len, voc_size = list(inputs.size())
        loss = self.loss_function(inputs.contiguous().view(-1, voc_size), targets.contiguous().view(-1))
        loss = loss.contiguous().view(bsize, seq_len).sum(dim=1)
        loss = loss if self.real_reduction == 'sum' else loss / lens.float()
        loss = (loss * conf).sum() if self.real_reduction == 'sum' else (loss * conf).sum() / conf.sum()
        return loss