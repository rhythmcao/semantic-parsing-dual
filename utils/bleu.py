#coding=utf8
import os, sys, nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_bleu_score(candidate_list, references_list, method=0, weights=(0.25, 0.25, 0.25, 0.25)):
    '''
        @args:
        if candidate_list is words list, e.g. ['which','flight']
            references_list is list of words list, e.g. [ ['which','flight'] , ['what','flight'] ]
            calculate bleu score of one sentence
        if candidate_list is list of words list, e.g. [ ['which','flight'] , ['when','to','flight'] ]
            references_list is list of list of words list, e.g.
            [   [ ['which','flight'] , ['what','flight'] ]   ,   [ ['when','to','flight'] , ['when','to','go'] ]   ]
            calculate bleu score of multiple sentences, a whole corpus
        method(int): chencherry smoothing methods choice
    '''
    chencherry = SmoothingFunction()
    if len(candidate_list) == 0:
        raise ValueError('[Error]: there is no candidate sentence!')
    if type(candidate_list[0]) == str:
        return sentence_bleu(
                    references_list,
                    candidate_list,
                    weights,
                    eval('chencherry.method' + str(method))
                )
    else:
        return corpus_bleu(
                    references_list,
                    candidate_list,
                    weights,
                    eval('chencherry.method' + str(method))
                )