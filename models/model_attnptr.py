#coding=utf8
from models.model_utils import lens2mask
from models.model_attn import AttnModel

class AttnPtrModel(AttnModel):

    def __init__(self, *args, **kargs):
        super(AttnPtrModel, self).__init__(*args, **kargs)
        self.copy = True
    
    def forward(self, src_inputs, src_lens, tgt_inputs, copy_tokens):
        """
            Used during training time.
        """
        enc_out, hidden_states = self.encoder(self.src_embed(src_inputs), src_lens)
        hidden_states = self.enc2dec(hidden_states)
        src_mask = lens2mask(src_lens)
        dec_out, _, copy_dist, gates = self.decoder(self.tgt_embed(tgt_inputs), hidden_states, enc_out, src_mask, copy_tokens)
        out = self.generator(dec_out, copy_dist, gates)
        return out

    def decode_one_step(self, ys, hidden_states, memory, src_mask, copy_tokens):
        dec_out, hidden_states, copy_dist, gates = self.decoder(self.tgt_embed(ys), hidden_states, memory, src_mask, copy_tokens)
        out = self.generator(dec_out, copy_dist, gates).squeeze(dim=1)
        return out, hidden_states
