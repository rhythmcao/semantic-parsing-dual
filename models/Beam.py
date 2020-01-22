from __future__ import division
import torch
from models import penalties
from utils.constants import *

class Beam(object):
    """
        Class for managing the internals of the beam search process.
        Takes care of beams, back pointers, and scores. (Revised from OpenNMT.)
        @args:
            size (int): beam size
            vocab (dict): contains indices of padding, beginning, and ending.
            min_length (int): minimum length to generate
            global_scorer (:obj:`GlobalScorer`)
            device (torch.device)
    """

    def __init__(self, size, vocab, min_length=2,
                 global_scorer=None, device=None):

        self.size = size
        self.device = device
        # The score for each translation on the beam.
        self.scores = torch.zeros(size, dtype=torch.float, device=self.device)

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.zeros(size, dtype=torch.long, device=self.device).fill_(vocab[PAD])]
        self.next_ys[0][0] = vocab[BOS]

        # Has EOS topped the beam yet.
        self._eos = vocab[EOS]
        self.eos_top = False

        # Other special symbols
        self._bos = vocab[BOS]
        self._pad = vocab[PAD]

        # Time and k pair for finished.
        self.finished = []

        # Information for global scoring.
        self.global_scorer = global_scorer

        # Minimum prediction length
        self.min_length = min_length

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs):
        """
        Given prob over words for every last beam `K x vocab` and update the beam.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        masks = torch.zeros(word_probs.size(), requires_grad=False, dtype=torch.float, device=self.device)
        masks[:, self._bos] = 1e20
        masks[:, self._pad] = 1e20 # prevent generate <s> <pad> symbol
        if cur_len < self.min_length:
            masks[:, self._eos] = 1e20 # prevent terminate too early
        word_probs = word_probs - masks

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + self.scores.unsqueeze(1)
            # Don't let EOS have children.
            masks = torch.zeros(beam_scores.size(), requires_grad=False, dtype=torch.float, device=self.device)
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    masks[i] = 1e20
            beam_scores = beam_scores - masks
        else:
            beam_scores = word_probs[0] # only start from <s>, not <pad>
        flat_beam_scores = beam_scores.contiguous().view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0, True, True)

        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))

        # check whether some sequence has terminated
        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores) # normalize score by length penalty
                rank_s, s = global_scores[i], self.scores[i]
                self.finished.append(([rank_s, s], len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.eos_top = True
        return self.done()

    def done(self):
        return self.eos_top and len(self.finished) >= self.size

    def sort_best(self):
        """
            Sort the current beam.
        """
        return torch.sort(self.scores, 0, True) # beam size
    
    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                rank_s, s = global_scores[i], self.scores[i]
                self.finished.append(([rank_s, s], len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0][0])
        scores = [sc[1] for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_temporary_hyp(self, k):
        """
            Get current hypotheses of rank k ( 0 <= rank <= beam_size-1 ). 
        """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return torch.stack(hyp[::-1])

    def get_hyp(self, timestep, k):
        """ 
            Walk back to construct the full hypothesis. 
            hyp contains </s> but does not contain <s>
            @return:
                hyp: LongTensor of size tgt_len
        """
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return torch.stack(hyp[::-1])

class GNMTGlobalScorer(object):
    """
    Re-ranking score revised from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
    """

    def __init__(self, alpha, len_penalty):
        self.alpha = alpha
        penalty_builder = penalties.PenaltyBuilder(len_penalty)
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam, logprobs, self.alpha)
        return normalized_probs
