#coding=utf8
"""
    Construct vocabulary for each dataset.
"""
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lexicon import Lexicon
from utils.constants import BOS, EOS, PAD, UNK
import operator

def read_data(path):
    ex_list = []
    with open(path, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line == '':
                continue
            q, lf = line.split('\t')
            q = [each.strip() for each in q.strip().split(' ') if each.strip() != '']
            lf = [each.strip() for each in lf.strip().split(' ') if each.strip() != '']
            ex_list.append((q, lf))
    return ex_list

def save_vocab(idx2word, vocab_path):
    with open(vocab_path, 'w') as f:
        for idx in range(len(idx2word)):
            f.write(idx2word[idx] + '\n')

def construct_vocab(input_seqs, mwf=1):
    '''
        Construct vocabulary given input_seqs
        @params:
            1. input_seqs: a list of seqs, e.g.
                [ ['what', 'flight'] , ['which', 'flight'] ]
            2. mwf: minimum word frequency
        @return:
            1. word2idx(dict)
            2. idx2word(dict)
    '''
    vocab, word2idx, idx2word = {}, {}, []
    for seq in input_seqs:
        if type(seq) in [tuple, list]:
            for word in seq:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        else:
            if seq not in vocab:
                vocab[seq] = 1
            else:
                vocab[seq] += 1
    
    # Discard those special tokens if already exist
    if PAD in vocab: del vocab[PAD]
    if UNK in vocab: del vocab[UNK]
    if BOS in vocab: del vocab[BOS]
    if EOS in vocab: del vocab[EOS]

    sorted_words = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    sorted_words = [x[0] for x in sorted_words if x[1] >= mwf]
    for word in sorted_words:
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word.append(word)
    return word2idx, idx2word

def main(args=sys.argv[1:]):
    """
        Construct vocabulary for each dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', help='dataset name')
    parser.add_argument('--mwf', type=int, default=1, help='minimum word frequency, if less than this int, not included')
    opt = parser.parse_args(args)
    all_dataset = [opt.dataset] if opt.dataset != 'all' else ['atis', 'geo', 'basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']

    for dataset in all_dataset:
        dirname = os.path.join('data', 'overnight') if dataset != 'atis' and dataset != 'geo' else os.path.join('data', dataset)
        file_path = os.path.join(dirname, dataset + '_train.tsv')
        word_vocab_path, lf_vocab_path, copy_vocab_path = os.path.join(dirname, dataset + '_vocab.word'), \
            os.path.join(dirname, dataset + '_vocab.lf'), os.path.join(dirname, dataset + '_vocab.copy')
        lexicon_words = sorted(list(Lexicon(dataset).seen_words))

        ex_list = read_data(file_path)
        questions, logical_forms = list(zip(*ex_list))
        _, id2word = construct_vocab(questions, mwf=opt.mwf)
        _, id2lf = construct_vocab(logical_forms, mwf=opt.mwf)
        _, id2copy = construct_vocab(lexicon_words, mwf=opt.mwf)
        save_vocab(id2word, word_vocab_path)
        save_vocab(id2lf, lf_vocab_path)
        save_vocab(id2copy, copy_vocab_path)

if __name__ == '__main__':

    main()
