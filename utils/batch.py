#coding=utf8
import sys, os, random
import torch
from utils.constants import *

def get_minibatch(data_list, vocab, task='semantic_parsing', data_index=None, index=0, batch_size=16, device=None, **kargs):
    index = index % len(data_list)
    batch_data_list = [data_list[idx] for idx in data_index[index: index + batch_size]]
    return BATCH_FUNC[task](batch_data_list, vocab, device, **kargs)

def get_minibatch_sp(ex_list, vocab, device, copy=False, **kargs):
    inputs = [ex.question for ex in ex_list]
    lens = [len(ex) for ex in inputs]
    lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)

    max_len = max(lens)
    padded_inputs = [sent + [PAD] * (max_len - len(sent)) for sent in inputs]
    inputs_idx = [[vocab.word2id[w] if w in vocab.word2id else vocab.word2id[UNK] for w in sent] for sent in padded_inputs]
    inputs_tensor = torch.tensor(inputs_idx, dtype=torch.long, device=device)

    outputs = [ex.logical_form for ex in ex_list]
    bos_eos_outputs = [[BOS] + sent + [EOS] for sent in outputs]
    out_lens = [len(each) for each in bos_eos_outputs]
    max_out_len = max(out_lens)
    padded_outputs = [sent + [PAD] * (max_out_len - len(sent)) for sent in bos_eos_outputs]
    outputs_idx = [[vocab.lf2id[w] if w in vocab.lf2id else vocab.lf2id[UNK] for w in sent] for sent in padded_outputs]
    outputs_tensor = torch.tensor(outputs_idx, dtype=torch.long, device=device)
    out_lens_tensor = torch.tensor(out_lens, dtype=torch.long, device=device)

    if copy: # pointer network need additional information
        mapped_inputs = [ex.mapped_question for ex in ex_list]
        oov_list, copy_inputs = [], []
        for sent in mapped_inputs:
            tmp_oov_list, tmp_copy_inputs = [], []
            for idx, word in enumerate(sent):
                if word not in vocab.lf2id and word not in tmp_oov_list and len(tmp_oov_list) < MAX_OOV_NUM:
                    tmp_oov_list.append(word)
                tmp_copy_inputs.append(
                    (
                        vocab.lf2id.get(word, vocab.lf2id[UNK]) if word in vocab.lf2id or word not in tmp_oov_list \
                        else len(vocab.lf2id) + tmp_oov_list.index(word) # tgt_vocab_size + oov_id
                    )
                )
            tmp_oov_list += [UNK] * (MAX_OOV_NUM - len(tmp_oov_list))
            oov_list.append(tmp_oov_list)
            copy_inputs.append(tmp_copy_inputs)

        copy_tokens = [
            torch.cat([
                torch.zeros(len(each), len(vocab.lf2id) + MAX_OOV_NUM, dtype=torch.float)\
                    .scatter_(-1, torch.tensor(each, dtype=torch.long).unsqueeze(-1), 1.0),
                torch.zeros(max_len - len(each), len(vocab.lf2id) + MAX_OOV_NUM, dtype=torch.float)
            ], dim=0)
            for each in copy_inputs
        ]
        copy_tokens = torch.stack(copy_tokens, dim=0).to(device) # bsize x src_len x (tgt_vocab + MAX_OOV_NUM)

        dec_outputs = [
            [
                len(vocab.lf2id) + oov_list[idx].index(tok)
                    if tok not in vocab.lf2id and tok in oov_list[idx] \
                    else vocab.lf2id.get(tok, vocab.lf2id[UNK])
                for tok in sent
            ] + [vocab.lf2id[PAD]] * (max_out_len - len(sent))
            for idx, sent in enumerate(bos_eos_outputs)
        ]
        dec_outputs_tensor = torch.tensor(dec_outputs, dtype=torch.long, device=device)
    else:
        dec_outputs_tensor, copy_tokens, oov_list = outputs_tensor, None, []

    return inputs_tensor, lens_tensor, outputs_tensor, dec_outputs_tensor, out_lens_tensor, copy_tokens, oov_list, (inputs, outputs)

def get_minibatch_qg(ex_list, vocab, device, copy=False, **kargs):
    raw_inputs = [ex.logical_form for ex in ex_list]
    inputs = [ex.mapped_logical_form for ex in ex_list] if copy else raw_inputs
    lens = [len(ex) for ex in inputs]
    lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)

    max_len = max(lens)
    padded_inputs = [sent + [PAD] * (max_len - len(sent)) for sent in inputs]
    inputs_idx = [[vocab.lf2id[w] if w in vocab.lf2id else vocab.lf2id[UNK] for w in sent] for sent in padded_inputs]
    inputs_tensor = torch.tensor(inputs_idx, dtype=torch.long, device=device)

    outputs = [ex.question for ex in ex_list]
    bos_eos_outputs = [[BOS] + sent + [EOS] for sent in outputs]
    out_lens = [len(each) for each in bos_eos_outputs]
    max_out_len = max(out_lens)
    padded_outputs = [sent + [PAD] * (max_out_len - len(sent)) for sent in bos_eos_outputs]
    outputs_idx = [[vocab.word2id[w] if w in vocab.word2id else vocab.word2id[UNK] for w in sent] for sent in padded_outputs]
    outputs_tensor = torch.tensor(outputs_idx, dtype=torch.long, device=device)
    out_lens_tensor = torch.tensor(out_lens, dtype=torch.long, device=device)

    if copy: # pointer network need additional information
        oov_list, copy_inputs = [], []
        for sent in inputs:
            tmp_oov_list, tmp_copy_inputs = [], []
            for idx, word in enumerate(sent):
                if word not in vocab.word2id and word not in tmp_oov_list and len(tmp_oov_list) < MAX_OOV_NUM:
                    tmp_oov_list.append(word)
                tmp_copy_inputs.append(
                    (
                        vocab.word2id.get(word, vocab.word2id[UNK]) if word in vocab.word2id or word not in tmp_oov_list \
                        else len(vocab.word2id) + tmp_oov_list.index(word) # tgt_vocab_size + oov_id
                    )
                )
            tmp_oov_list += [UNK] * (MAX_OOV_NUM - len(tmp_oov_list))
            oov_list.append(tmp_oov_list)
            copy_inputs.append(tmp_copy_inputs)

        copy_tokens = [
            torch.cat([
                torch.zeros(len(each), len(vocab.word2id) + MAX_OOV_NUM, dtype=torch.float)\
                    .scatter_(-1, torch.tensor(each, dtype=torch.long).unsqueeze(-1), 1.0),
                torch.zeros(max_len - len(each), len(vocab.word2id) + MAX_OOV_NUM, dtype=torch.float)
            ], dim=0)
            for each in copy_inputs
        ]
        copy_tokens = torch.stack(copy_tokens, dim=0).to(device) # bsize x src_len x (tgt_vocab + MAX_OOV_NUM)

        dec_outputs = [
            [
                len(vocab.word2id) + oov_list[idx].index(tok)
                    if tok not in vocab.word2id and tok in oov_list[idx] \
                    else vocab.word2id.get(tok, vocab.word2id[UNK])
                for tok in sent
            ] + [vocab.word2id[PAD]] * (max_out_len - len(sent))
            for idx, sent in enumerate(bos_eos_outputs)
        ]
        dec_outputs_tensor = torch.tensor(dec_outputs, dtype=torch.long, device=device)
    else:
        dec_outputs_tensor, copy_tokens, oov_list = outputs_tensor, None, []

    return inputs_tensor, lens_tensor, outputs_tensor, dec_outputs_tensor, out_lens_tensor, copy_tokens, oov_list, (raw_inputs, outputs)

def get_minibatch_unlabeled_sp(ex_list, vocab, device, copy=False, **kargs):
    inputs = [ex.question for ex in ex_list]
    lens = [len(ex) for ex in inputs]
    lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)

    max_len = max(lens)
    padded_inputs = [sent + [PAD] * (max_len - len(sent)) for sent in inputs]
    inputs_idx = [[vocab.word2id[w] if w in vocab.word2id else vocab.word2id[UNK] for w in sent] for sent in padded_inputs]
    inputs_tensor = torch.tensor(inputs_idx, dtype=torch.long, device=device)

    if copy: # pointer network need additional information
        mapped_inputs = [ex.mapped_question for ex in ex_list]
        oov_list, copy_inputs = [], []
        for sent in mapped_inputs:
            tmp_oov_list, tmp_copy_inputs = [], []
            for idx, word in enumerate(sent):
                if word not in vocab.lf2id and word not in tmp_oov_list and len(tmp_oov_list) < MAX_OOV_NUM:
                    tmp_oov_list.append(word)
                tmp_copy_inputs.append(
                    (
                        vocab.lf2id.get(word, vocab.lf2id[UNK]) if word in vocab.lf2id or word not in tmp_oov_list \
                        else len(vocab.lf2id) + tmp_oov_list.index(word) # tgt_vocab_size + oov_id
                    )
                )
            tmp_oov_list += [UNK] * (MAX_OOV_NUM - len(tmp_oov_list))
            oov_list.append(tmp_oov_list)
            copy_inputs.append(tmp_copy_inputs)

        copy_tokens = [
            torch.cat([
                torch.zeros(len(each), len(vocab.lf2id) + MAX_OOV_NUM, dtype=torch.float)\
                    .scatter_(-1, torch.tensor(each, dtype=torch.long).unsqueeze(-1), 1.0),
                torch.zeros(max_len - len(each), len(vocab.lf2id) + MAX_OOV_NUM, dtype=torch.float)
            ], dim=0)
            for each in copy_inputs
        ]
        copy_tokens = torch.stack(copy_tokens, dim=0).to(device) # bsize x src_len x (tgt_vocab + MAX_OOV_NUM)
    else:
        copy_tokens, oov_list = None, []

    return inputs_tensor, lens_tensor, copy_tokens, oov_list, inputs

def get_minibatch_unlabeled_qg(ex_list, vocab, device, copy=False, **kargs):
    raw_inputs = [ex.logical_form for ex in ex_list]
    inputs = [ex.mapped_logical_form for ex in ex_list] if copy else raw_inputs
    lens = [len(ex) for ex in inputs]
    lens_tensor = torch.tensor(lens, dtype=torch.long, device=device)

    max_len = max(lens)
    padded_inputs = [sent + [PAD] * (max_len - len(sent)) for sent in inputs]
    inputs_idx = [[vocab.lf2id[w] if w in vocab.lf2id else vocab.lf2id[UNK] for w in sent] for sent in padded_inputs]
    inputs_tensor = torch.tensor(inputs_idx, dtype=torch.long, device=device)

    if copy: # pointer network need additional information
        oov_list, copy_inputs = [], []
        for sent in inputs:
            tmp_oov_list, tmp_copy_inputs = [], []
            for idx, word in enumerate(sent):
                if word not in vocab.word2id and word not in tmp_oov_list and len(tmp_oov_list) < MAX_OOV_NUM:
                    tmp_oov_list.append(word)
                tmp_copy_inputs.append(
                    (
                        vocab.word2id.get(word, vocab.word2id[UNK]) if word in vocab.word2id or word not in tmp_oov_list \
                        else len(vocab.word2id) + tmp_oov_list.index(word) # tgt_vocab_size + oov_id
                    )
                )
            tmp_oov_list += [UNK] * (MAX_OOV_NUM - len(tmp_oov_list))
            oov_list.append(tmp_oov_list)
            copy_inputs.append(tmp_copy_inputs)

        copy_tokens = [
            torch.cat([
                torch.zeros(len(each), len(vocab.word2id) + MAX_OOV_NUM, dtype=torch.float)\
                    .scatter_(-1, torch.tensor(each, dtype=torch.long).unsqueeze(-1), 1.0),
                torch.zeros(max_len - len(each), len(vocab.word2id) + MAX_OOV_NUM, dtype=torch.float)
            ], dim=0)
            for each in copy_inputs
        ]
        copy_tokens = torch.stack(copy_tokens, dim=0).to(device) # bsize x src_len x (tgt_vocab + MAX_OOV_NUM)
    else:
        copy_tokens, oov_list = None, []

    return inputs_tensor, lens_tensor, copy_tokens, oov_list, raw_inputs

def get_minibatch_pseudo_sp(ex_list, vocab, device, copy=False, **kargs):
    inputs, lens, outputs, dec_outputs, out_lens, copy_tokens, _, _ = get_minibatch_sp(ex_list, vocab, device, copy=copy, **kargs)
    conf = torch.tensor([ex.conf for ex in ex_list], dtype=torch.float, device=device)
    return inputs, lens, outputs, dec_outputs, out_lens, copy_tokens, conf

def get_minibatch_pseudo_qg(ex_list, vocab, device, copy=False, **kargs):
    inputs, lens, outputs, dec_outputs, out_lens, copy_tokens, _, _ = get_minibatch_qg(ex_list, vocab, device, copy=copy, **kargs)
    conf = torch.tensor([ex.conf for ex in ex_list], dtype=torch.float, device=device)
    return inputs, lens, outputs, dec_outputs, out_lens, copy_tokens, conf

def get_minibatch_lm(ex_list, vocab, device, side='question', **kargs):
    if side == 'question':
        word2id = vocab.word2id
        inputs = [ex.question for ex in ex_list]
    else:
        word2id = vocab.lf2id
        inputs = [ex.logical_form for ex in ex_list]
    bos_eos_inputs = [[BOS] + sent + [EOS] for sent in inputs]
    lens = [len(each) for each in bos_eos_inputs]
    max_len = max(lens)
    padded_inputs = [sent + [PAD] * (max_len - len(sent)) for sent in bos_eos_inputs]
    inputs_idx = [[word2id[w] if w in word2id else word2id[UNK] for w in sent] for sent in padded_inputs]
    inputs_tensor = torch.tensor(inputs_idx, dtype=torch.long, device=device)
    lens = torch.tensor(lens, dtype=torch.long, device=device)
    return inputs_tensor, lens, inputs

BATCH_FUNC = {
    "semantic_parsing": get_minibatch_sp,
    "question_generation": get_minibatch_qg,
    "unlabeled_semantic_parsing": get_minibatch_unlabeled_sp,
    "unlabeled_question_generation": get_minibatch_unlabeled_qg,
    "pseudo_semantic_parsing": get_minibatch_pseudo_sp,
    "pseudo_question_generation": get_minibatch_pseudo_qg,
    "language_model": get_minibatch_lm
}
