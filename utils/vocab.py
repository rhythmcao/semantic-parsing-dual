#coding=utf8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.constants import BOS, EOS, PAD, UNK

class Vocab():

    def __init__(self, dataset, task='semantic_parsing', copy=False):
        super(Vocab, self).__init__()
        self.dataset = dataset
        dirname = os.path.join('data', 'overnight') if dataset != 'atis' and dataset != 'geo' else os.path.join('data', dataset)
        word_path = os.path.join(dirname, dataset + '_vocab.word')
        lf_path = os.path.join(dirname, dataset + '_vocab.lf')
        copy_path = os.path.join(dirname, dataset + '_vocab.copy')
        if task == 'semantic_parsing':
            self.word2id, self.id2word = self.read_vocab(word_path, bos_eos=False)
            self.lf2id, self.id2lf = self.read_vocab(lf_path, bos_eos=True)
        elif task == 'question_generation':
            self.word2id, self.id2word = self.read_vocab(word_path, bos_eos=True)
            if copy:
                self.lf2id, self.id2lf = self.read_vocab(lf_path, copy_path, bos_eos=False)
            else:
                self.lf2id, self.id2lf = self.read_vocab(lf_path, bos_eos=False)
        elif task == 'language_model':
            self.word2id, self.id2word = self.read_vocab(word_path, bos_eos=True)
            self.lf2id, self.id2lf = self.read_vocab(lf_path, bos_eos=True)
        else:
            raise ValueError('[Error]: unknown task !')

    def read_vocab(self, *args, bos_eos=True, pad=True, unk=True, separator=' : '):
        word2idx, idx2word = {}, []
        if pad:
            word2idx[PAD] = len(word2idx)
            idx2word.append(PAD)
        if unk:
            word2idx[UNK] = len(word2idx)
            idx2word.append(UNK)
        if bos_eos:
            word2idx[BOS] = len(word2idx)
            idx2word.append(BOS)
            word2idx[EOS] = len(word2idx)
            idx2word.append(EOS)
        for vocab_path in args:
            with open(vocab_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == '':
                        continue
                    if separator in line:
                        word, _ = line.split(separator)
                    else:
                        word = line
                    idx = len(word2idx)
                    if word not in word2idx:
                        word2idx[word] = idx
                        idx2word.append(word)
        return word2idx, idx2word