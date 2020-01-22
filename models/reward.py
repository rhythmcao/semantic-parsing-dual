#coding=utf8
from utils.constants import *
from utils.example import Example
from models.model_utils import lens2mask
import numpy as np
import torch

class RewardModel():

    def __init__(self, dataset, qlm, lflm, lm_vocab, sp_device='cpu', qg_device='cpu'):
        super(RewardModel, self).__init__()
        self.dataset = dataset
        self.qlm = qlm.to(sp_device)
        self.lflm = lflm.to(qg_device)
        self.vocab = lm_vocab
        self.sp_device = sp_device
        self.qg_device = qg_device

    def forward(self, *args, choice='sp_val'):
        if choice == 'sp_val':
            return self.sp_validity_reward(*args)
        elif choice == 'qg_val':
            return self.qg_validity_reward(*args)
        elif 'rec' in choice:
            return self.reconstruction_reward(*args)
        else:
            raise ValueError('[Error]: unknown reward choice !')

    def sp_validity_reward(self, lf_list):
        # calculate logical form language model length normalized log probability
        input_idxs = [[self.vocab.lf2id[BOS]] + [self.vocab.lf2id[word] if word in self.vocab.lf2id else self.vocab.lf2id[UNK] for word in sent] + [self.vocab.word2id[EOS]] for sent in lf_list]
        lens = [len(each) for each in input_idxs]
        max_len = max(lens)
        input_idxs = [sent + [self.vocab.lf2id[PAD]] * (max_len - len(sent)) for sent in input_idxs]
        input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.qg_device)
        lens = torch.tensor(lens, dtype=torch.long, device=self.qg_device)
        self.lflm.eval()
        with torch.no_grad():
            logprob = self.lflm.sent_logprobability(input_tensor, lens).cpu()
        # grammar check
        domain = Example.domain
        ans = domain.is_valid(domain.obtain_denotations(domain.normalize(lf_list)))
        grammar = torch.tensor(ans, dtype=torch.float, requires_grad=False)
        val_reward = 0.5 * logprob + 0.5 * grammar
        return val_reward

    def qg_validity_reward(self, utterances):
        # calculate language model length normalized log probability
        input_idxs = [[self.vocab.word2id[BOS]] + [self.vocab.word2id[word] if word in self.vocab.word2id else self.vocab.word2id[UNK] for word in sent] + [self.vocab.word2id[EOS]] for sent in utterances]
        lens = [len(each) for each in input_idxs]
        max_len = max(lens)
        input_idxs = [sent + [self.vocab.word2id[PAD]] * (max_len - len(sent)) for sent in input_idxs]
        input_tensor = torch.tensor(input_idxs, dtype=torch.long, device=self.sp_device)
        lens = torch.tensor(lens, dtype=torch.long, device=self.sp_device)
        self.qlm.eval()
        with torch.no_grad():
            logprob = self.qlm.sent_logprobability(input_tensor, lens).cpu()
        return logprob

    def reconstruction_reward(self, logscores, references, lens):
        """
            logscores: bsize x max_out_len x vocab_size[ + MAX_OOV_NUM]
            references: bsize x max_out_len
            lens: len for each sample
        """
        mask = lens2mask(lens)
        pick_score = torch.gather(logscores, dim=-1, index=references.unsqueeze(dim=-1)).squeeze(dim=-1)
        masked_score = mask.float() * pick_score
        reward = masked_score.sum(dim=1)
        return reward

    def __call__(self, *args, **kargs):
        return self.forward(*args, **kargs)
