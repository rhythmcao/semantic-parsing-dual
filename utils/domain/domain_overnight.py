#coding=utf8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.domain.domain_base import Domain
import tempfile
import subprocess
import re

class OvernightDomain(Domain):

    def __init__(self, dataset):
        self.dataset = dataset
        self.denotation = True

    def normalize(self, lf_list):
        lf_list = [' '.join(lf) for lf in lf_list]

        def format_overnight(lf):
            replacements = [
                ('(', ' ( '), # make sure ( and ) must have blank space around
                (')', ' ) '),
                ('! ', '!'),
                ('SW', 'edu.stanford.nlp.sempre.overnight.SimpleWorld'),
            ]
            for a, b in replacements:
                lf = lf.replace(a, b)
            # remove redundant blank spaces
            lf = re.sub(' +', ' ', lf)
            return lf.strip()

        return [format_overnight(lf) for lf in lf_list]

    def obtain_denotations(self, lf_list):
        tf = tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.examples')
        for line in lf_list:
            tf.write(line + '\n')
        tf.flush()
        msg = subprocess.check_output(['evaluator/overnight', self.dataset, tf.name])
        msg = msg.decode('utf8')
        tf.close()
        denotations = [
            line.split('\t')[1] for line in msg.split('\n')
            if line.startswith('targetValue\t')
        ]
        return denotations

    def is_valid(self, ans_list):
        return list(map(lambda ans: 0.0 if 'BADJAVA' in ans or 'ERROR' in ans or ans == 'null' else 1.0, ans_list)) 
