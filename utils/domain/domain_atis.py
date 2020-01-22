#coding=utf8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.domain.domain_base import Domain
from utils.domain.atis_evaluator import ATISEvaluator

class ATISDomain(Domain):

    def __init__(self):

        self.dataset = 'atis'
        self.denotation = False
        self.evaluator = ATISEvaluator()

    def to_lisp_tree(self, toks):
        '''
            input(list): ['lambda', '$0', 'e', '(', 'flight', '$0', ')']
            return(recursive list): ['lambda', '$0', 'e', ['flight', '$0']]
        '''
        def recurse(i):
            if toks[i] == '(':
                subtrees = []
                j = i + 1
                while True:
                    subtree, j = recurse(j)
                    subtrees.append(subtree)
                    if toks[j] == ')':
                        return subtrees, j + 1
            else:
                return toks[i], i+1

        try:
            lisp_tree, final_ind = recurse(0)
            return lisp_tree
        except Exception as e:
            return None

    def sort_args(self, lf):
        lisp_tree = self.to_lisp_tree(lf)
        if lisp_tree is None: # failed to convert to logical tree
            return ' '.join(lf)

        def recurse(node): # Post-order traversal, sort and/or subtrees
            if isinstance(node, str):
                return
            for child in node:
                recurse(child)
            if node[0] in ('_and', '_or', 'and', 'or'):
                node[1:] = sorted(node[1:], key=lambda x: str(x))

        recurse(lisp_tree)

        def tree_to_str(node):
            if isinstance(node, str):
                return node
            else:
                return '( %s )' % ' '.join(tree_to_str(child) for child in node)

        return tree_to_str(lisp_tree)

    def normalize(self, lf_list):
        sorted_lf_list = [self.sort_args(lf) for lf in lf_list]
        return sorted_lf_list

    def is_valid(self, ans_list):

        def bracket_matching(lf):
            left = 0
            for each in lf:
                if each == '(':
                    left += 1
                elif each == ')':
                    left -= 1
                if left < 0:
                    return 0.0
            return 1.0 if left == 0 else 0.0

        ans_list = [[i.strip() for i in lf.split(' ') if i.strip() != ''] for lf in ans_list]
        bracket_signal = list(map(bracket_matching, ans_list))
        lisp_trees = [self.to_lisp_tree(each) if bracket_signal[idx] == 1.0 else None for idx, each in enumerate(ans_list)]
        type_consistency = [self.evaluator.eval(each) if each is not None else 0.0 for each in lisp_trees]
        return list(map(lambda x, y: (0.5 if x is not None else 0.0) + 0.5 * y, lisp_trees, type_consistency))

if __name__ =='__main__':

    d = ATISDomain()

    lfs = [
       'al0',
       '( count $0 ( and ( flight $0 ) ( airline $0 al0 ) ) )',
       '( lambda $0 e ( and ( flight $0 ) ( approx_departure_time $0 ti0 ) ( during_day $0 afternoon:pd ) ( from $0 ci0 ) ( to $0 ci1 ) ) )',
       '( min $0 ( exists $1 ( and ( from $1 ci0 ) ( to $1 ci1 ) ( round_trip $1 ) ( = ( fare $1 ) $0 ) ) ) )',
       '( sum $0 ( and ( aircraft $0 ) ( airline $0 al0 ) ) ( capacity $0 ) )',
       '( argmin $0 ( and ( flight $0 ) ( from $0 ci0 ) ( to $0 ci1 ) ( day $0 da0 ) ) ( departure_time $0 ) )',
    ]
    lfs = [[i.strip() for i in lf.split(' ') if i.strip() != ''] for lf in lfs]
    lisp_trees = [d.to_lisp_tree(lf) for lf in lfs]
    for lf in lisp_trees:
        print('\nInput:', lf)
        try:
            t, v = d.evaluator.check_type_consistency(lf, variables={})
            print('Return type: %s' % (t))
            print('Variables: %s' % (v))
        except Exception as e:
            print('Type inconsistent !')
            print(e)

    exit(0)

    def read_lfs(fp, idx=0):
        lfs = []
        with open(fp, 'r') as infile:
            for line in infile:
                line = line.strip()
                if line == '': continue
                q, lf = line.split('\t')
                lf = [i.strip() for i in lf.split(' ') if i.strip() != '']
                lfs.append((q, lf, idx))
                idx += 1
        return lfs

    dataset = read_lfs('data/atis/atis_train.tsv') + read_lfs('data/atis/atis_dev.tsv') + \
            read_lfs('data/atis/atis_test.tsv') + read_lfs('data/atis/atis_extra.tsv')
    count = 0
    for (q, lf, i) in dataset:
        lisp_tree = d.to_lisp_tree(lf)
        score = d.evaluator.eval(lisp_tree)
        if score == 0.0:
            count += 1
            print('The %d-th sample is type inconsistent:' % (i + 1))
            print(q)
            print(' '.join(lf))
            print('')
    print('Total error: %d' % (count))
