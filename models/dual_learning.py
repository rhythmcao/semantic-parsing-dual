#coding=utf8
import os, sys
import torch
import torch.nn as nn
from utils.example import Example
from utils.constants import PAD, UNK, BOS, EOS
from utils.batch import get_minibatch_sp, get_minibatch_qg

class DualLearning(nn.Module):

    def __init__(self, sp_model, qg_model, reward_model, sp_vocab, qg_vocab,
        alpha=0.5, beta=0.5, sample=5, reduction='sum', sp_device=None, qg_device=None, **kargs):
        """
            @args:
                1. alpha: reward for cycle starting from sp_reward = val_reward * alpha + rec_reward * (1 - alpha)
                2. beta: reward for cycle starting from qg_reward = val_reward * beta + rec_reward * (1 - beta)
                3. sample: beam search and sample size for training in dual learning cycles
        """
        super(DualLearning, self).__init__()
        self.sp_device = sp_device
        self.qg_device = qg_device
        self.sp_model = sp_model.to(self.sp_device)
        self.qg_model = qg_model.to(self.qg_device)
        self.reward_model = reward_model
        self.alpha, self.beta, self.sample = alpha, beta, sample
        self.reduction = reduction
        self.sp_vocab = sp_vocab
        self.qg_vocab = qg_vocab

    def forward(self, *args, start_from='semantic_parsing', **kargs):
        """
            @args:
                *args(tensors): positional arguments for semantic parsing or question generation
                start_from(enum): semantic_parsing or question_generation
        """
        if start_from == 'semantic_parsing':
            return self.cycle_start_from_sp(*args, **kargs)
        elif start_from == 'question_generation':
            return self.cycle_start_from_qg(*args, **kargs)
        else:
            raise ValueError('[Error]: dual learning cycle with unknown starting point !')

    def cycle_start_from_sp(self, inputs, lens, copy_tokens, oov_list, raw_in):
        domain = Example.domain
        # primal model
        results = self.sp_model.decode_batch(inputs, lens, self.sp_vocab.lf2id, copy_tokens, self.sample, self.sample)
        predictions, sp_scores = results['predictions'], results['scores']
        predictions = [idx for each in predictions for idx in each]
        predictions = domain.reverse(predictions, self.sp_vocab.id2lf, oov_list=oov_list)
        raw_in = [each for each in raw_in for _ in range(self.sample)] # repeat sample times

        # calculate validity reward
        sp_val_reward = self.reward_model(predictions, choice='sp_val').contiguous().view(-1, self.sample)
        baseline = sp_val_reward.mean(dim=-1, keepdim=True)
        sp_val_reward -= baseline

        # dual model
        qg_inputs, qg_lens, qg_dec_inputs, qg_dec_outputs, qg_out_lens, qg_copy_tokens = \
            self.sp2qg(predictions, raw_in, vocab=self.qg_vocab, device=self.qg_device)
        logscore = self.qg_model(qg_inputs, qg_lens, qg_dec_inputs[:, :-1], qg_copy_tokens)

        # calculate reconstruction reward
        rec_reward = self.reward_model(logscore, qg_dec_outputs[:, 1:], qg_out_lens - 1, choice='sp_rec').contiguous().view(-1, self.sample)
        sp_rec_reward = rec_reward.detach().cpu()
        baseline = sp_rec_reward.mean(dim=-1, keepdim=True)
        sp_rec_reward = sp_rec_reward - baseline

        total_reward = self.alpha * sp_val_reward + (1 - self.alpha) * sp_rec_reward
        sp_loss = - torch.mean(total_reward.to(self.sp_device) * sp_scores, dim=1)
        sp_loss = torch.sum(sp_loss) if self.reduction == 'sum' else torch.mean(sp_loss)
        qg_loss = - torch.mean((1 - self.alpha) * rec_reward, dim=1)
        qg_loss = torch.sum(qg_loss) if self.reduction == 'sum' else torch.mean(qg_loss)
        return sp_loss, qg_loss

    def sp2qg(self, lf_list, utterances, vocab, device):
        ex_list = [Example(' '.join(sent), ' '.join(lf)) for sent, lf in zip(utterances, lf_list)]
        inputs, lens, dec_inputs, dec_outputs, out_lens, copy_tokens, _, _ = \
            get_minibatch_qg(ex_list, vocab, device, copy=self.qg_model.copy)
        return inputs, lens, dec_inputs, dec_outputs, out_lens, copy_tokens

    def cycle_start_from_qg(self, inputs, lens, copy_tokens, oov_list, raw_in):
        domain = Example.domain
        # primal model
        results = self.qg_model.decode_batch(inputs, lens, self.qg_vocab.word2id, copy_tokens, self.sample, self.sample)
        predictions, qg_scores = results['predictions'], results['scores']
        predictions = [idx for each in predictions for idx in each]
        predictions = domain.reverse(predictions, self.qg_vocab.id2word, oov_list=oov_list)
        raw_in = [each for each in raw_in for _ in range(self.sample)] # repeat sample times

        # calculate validity reward
        qg_val_reward = self.reward_model(predictions, choice='qg_val').contiguous().view(-1, self.sample)
        baseline = qg_val_reward.mean(dim=-1, keepdim=True)
        qg_val_reward -= baseline

        # dual model
        sp_inputs, sp_lens, sp_dec_inputs, sp_dec_outputs, sp_out_lens, sp_copy_tokens = \
            self.qg2sp(predictions, raw_in, self.sp_vocab, self.sp_device)
        logscore = self.sp_model(sp_inputs, sp_lens, sp_dec_inputs[:, :-1], sp_copy_tokens)

        # calculate reconstruction reward
        rec_reward = self.reward_model(logscore, sp_dec_outputs[:, 1:], sp_out_lens - 1, choice='qg_rec').contiguous().view(-1, self.sample)
        qg_rec_reward = rec_reward.detach().cpu()
        baseline = qg_rec_reward.mean(dim=-1, keepdim=True)
        qg_rec_reward = qg_rec_reward - baseline

        total_reward = self.beta * qg_val_reward + (1 - self.beta) * qg_rec_reward
        qg_loss = - torch.mean(total_reward.to(self.qg_device) * qg_scores, dim=1)
        qg_loss = torch.sum(qg_loss) if self.reduction == 'sum' else torch.mean(qg_loss)
        sp_loss = - torch.mean((1 - self.beta) * rec_reward, dim=1)
        sp_loss = torch.sum(sp_loss) if self.reduction == 'sum' else torch.mean(sp_loss)
        return sp_loss, qg_loss

    def qg2sp(self, utterances, lf_list, vocab, device):
        ex_list = [Example(' '.join(sent), ' '.join(lf)) for sent, lf in zip(utterances, lf_list)]
        inputs, lens, dec_inputs, dec_outputs, out_lens, copy_tokens, _, _ = \
            get_minibatch_sp(ex_list, vocab, device, copy=self.sp_model.copy)
        return inputs, lens, dec_inputs, dec_outputs, out_lens, copy_tokens

    def decode_batch(self, *args, task='semantic_parsing', **kargs):
        if task == 'semantic_parsing':
            return self.sp_model.decode_batch(*args, **kargs)
        elif task == 'question_generation':
            return self.qg_model.decode_batch(*args, **kargs)
        else:
            raise ValueError('[Error]: unknown task name !')

    def pad_embedding_grad_zero(self):
        self.sp_model.pad_embedding_grad_zero()
        self.qg_model.pad_embedding_grad_zero()

    def load_model(self, sp_load_dir=None, qg_load_dir=None):
        if sp_load_dir is not None:
            self.sp_model.load_model(sp_load_dir)
        if qg_load_dir is not None:
            self.qg_model.load_model(qg_load_dir)

    def save_model(self, sp_save_dir=None, qg_save_dir=None):
        if sp_save_dir is not None:
            self.sp_model.save_model(sp_save_dir)
        if qg_save_dir is not None:
            self.qg_model.save_model(qg_save_dir)
