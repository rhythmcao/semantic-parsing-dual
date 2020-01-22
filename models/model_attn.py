#coding=utf8
import torch
import torch.nn as nn
from utils.constants import BOS, EOS, MAX_DECODE_LENGTH
from models.model_utils import tile, lens2mask
from models.Beam import Beam, GNMTGlobalScorer
from models.encoder_decoder import EncoderDecoder

class AttnModel(EncoderDecoder):

    def __init__(self, *args, **kargs):
        super(AttnModel, self).__init__(*args, **kargs)
        self.copy = False

    """
        We use copy_tokens, just to be compatible with Attention Pointer Model
    """
    def forward(self, src_inputs, src_lens, tgt_inputs, copy_tokens=None):
        """
            Used during training time.
        """
        enc_out, hidden_states = self.encoder(self.src_embed(src_inputs), src_lens)
        hidden_states = self.enc2dec(hidden_states)
        src_mask = lens2mask(src_lens)
        dec_out, _ = self.decoder(self.tgt_embed(tgt_inputs), hidden_states, enc_out, src_mask, copy_tokens)
        out = self.generator(dec_out)
        return out

    def decode_batch(self, src_inputs, src_lens, vocab, copy_tokens=None,
            beam_size=5, n_best=1, alpha=0.6, length_pen='avg'):
        enc_out, hidden_states = self.encoder(self.src_embed(src_inputs), src_lens)
        hidden_states = self.enc2dec(hidden_states)
        src_mask = lens2mask(src_lens)
        if beam_size == 1:
            return self.decode_greed(hidden_states, enc_out, src_mask, vocab, copy_tokens)
        else:
            return self.decode_beam_search(hidden_states, enc_out, src_mask, vocab, copy_tokens, 
                beam_size=beam_size, n_best=n_best, alpha=alpha, length_pen=length_pen)

    def decode_greed(self, hidden_states, memory, src_mask, vocab, copy_tokens=None):
        """
            hidden_states: hidden_states from encoder
            memory: encoder output, bsize x src_len x enc_dim
            src_mask: ByteTensor, bsize x max_src_len
            vocab: tgt word2idx dict containing BOS, EOS
        """
        results = {"scores":[], "predictions":[]}
        
        # first target token is BOS
        batches = memory.size(0)
        ys = torch.ones(batches, 1, dtype=torch.long).fill_(vocab[BOS]).to(memory.device)
        # record whether each sample is finished
        all_done = torch.tensor([False] * batches, dtype=torch.uint8, device=memory.device)
        scores = torch.zeros(batches, 1, dtype=torch.float, device=memory.device)
        predictions = [[] for i in range(batches)]

        for i in range(MAX_DECODE_LENGTH):
            logprob, hidden_states = self.decode_one_step(ys, hidden_states, memory, src_mask, copy_tokens)
            maxprob, ys = torch.max(logprob, dim=1, keepdim=True)
            for i in range(batches):
                if not all_done[i]:
                    scores[i] += maxprob[i]
                    predictions[i].append(ys[i])
            done = ys.squeeze(dim=1) == vocab[EOS]
            all_done |= done
            if all_done.all():
                break
        results["predictions"], results["scores"] = [[torch.cat(pred).tolist()] for pred in predictions], scores
        return results
    
    def decode_one_step(self, ys, hidden_states, memory, src_mask, copy_tokens=None):
        """
            ys: bsize x 1
        """
        dec_out, hidden_states = self.decoder(self.tgt_embed(ys), hidden_states, memory, src_mask, copy_tokens)
        out = self.generator(dec_out)
        return out.squeeze(dim=1), hidden_states

    def decode_beam_search(self, hidden_states, memory, src_mask, vocab, copy_tokens=None, 
            beam_size=5, n_best=1, alpha=0.6, length_pen='avg'):
        """
            Beam search decoding
        """
        results = {"scores":[], "predictions":[]}
        
        # Construct beams, we donot use stepwise coverage penalty nor ngrams block
        remaining_sents = memory.size(0)
        global_scorer = GNMTGlobalScorer(alpha, length_pen)
        beam = [ Beam(beam_size, vocab, global_scorer=global_scorer, device=memory.device)
                for _ in range(remaining_sents) ]

        # repeat beam_size times
        memory, src_mask, copy_tokens = tile([memory, src_mask, copy_tokens], beam_size, dim=0)
        hidden_states = tile(hidden_states, beam_size, dim=1)
        h_c = type(hidden_states) in [list, tuple]
        batch_idx = list(range(remaining_sents))

        for i in range(MAX_DECODE_LENGTH):
            # (a) construct beamsize * remaining_sents next words
            ys = torch.stack([b.get_current_state() for b in beam if not b.done()]).contiguous().view(-1,1)

            # (b) pass through the decoder network
            out, hidden_states = self.decode_one_step(ys, hidden_states, memory, src_mask, copy_tokens)
            out = out.contiguous().view(remaining_sents, beam_size, -1)

            # (c) advance each beam
            active, select_indices_array = [], []
            # Loop over the remaining_batch number of beam
            for b in range(remaining_sents):
                idx = batch_idx[b] # idx represent the original order in minibatch_size
                beam[idx].advance(out[b])
                if not beam[idx].done():
                    active.append((idx, b))
                select_indices_array.append(beam[idx].get_current_origin() + b * beam_size)

            # (d) update hidden_states history
            select_indices_array = torch.cat(select_indices_array, dim=0)
            if h_c:
                hidden_states = (hidden_states[0].index_select(1, select_indices_array), hidden_states[1].index_select(1, select_indices_array))
            else:
                hidden_states = hidden_states.index_select(1, select_indices_array)
            
            if not active:
                break

            # (e) reserve un-finished batches
            active_idx = torch.tensor([item[1] for item in active], dtype=torch.long, device=memory.device) # original order in remaining batch
            batch_idx = { idx: item[0] for idx, item in enumerate(active) } # order for next remaining batch

            def update_active(t):
                if t is None: return t
                t_reshape = t.contiguous().view(remaining_sents, beam_size, -1)
                new_size = list(t.size())
                new_size[0] = -1
                return t_reshape.index_select(0, active_idx).view(*new_size)

            if h_c:
                hidden_states = (
                    update_active(hidden_states[0].transpose(0, 1)).transpose(0, 1).contiguous(),
                    update_active(hidden_states[1].transpose(0, 1)).transpose(0, 1).contiguous()
                )
            else:
                hidden_states = update_active(hidden_states.transpose(0, 1)).transpose(0, 1).contiguous()
            memory = update_active(memory)
            src_mask = update_active(src_mask)
            copy_tokens = update_active(copy_tokens)
            remaining_sents = len(active)

        for b in beam:
            scores, ks = b.sort_finished(minimum=n_best)
            hyps = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp = b.get_hyp(times, k)
                hyps.append(hyp.tolist()) # hyp contains </s> but does not contain <s>
            results["predictions"].append(hyps) # batch list of variable_tgt_len
            results["scores"].append(torch.stack(scores)[:n_best]) # list of [n_best], torch.FloatTensor
        results["scores"] = torch.stack(results["scores"])
        return results
