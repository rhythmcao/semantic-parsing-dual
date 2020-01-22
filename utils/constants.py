#coding=utf8
BOS = '<s>'
EOS = '</s>'
PAD = '<pad>'
UNK = '<unk>'
MAX_DECODE_LENGTH = 100
MAX_OOV_NUM = 50
VECTORCACHE = lambda emb_dim: 'data/.cache/glove.6B.' + str(emb_dim) + 'd.txt'