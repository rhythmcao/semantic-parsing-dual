#coding=utf8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.domain.domain_base import Domain
import tempfile
import subprocess
import re

class GEODomain(Domain):

    def __init__(self):
        self.dataset = 'geo'
        self.denotation = True

    def normalize(self, lf_list):
        def format_geo(lf):
            """
                lf is token list
                1. for entity longer than one word, add double quotes
                2. remove underline _ for predicates
                3. remove unnecessary space, except spaces in entities
            """
            toks, quoted_toks, in_quotes = [], [], False
            for t in lf:
                if in_quotes:
                    if t == "'": # entity ending
                        toks.append('"%s"' % ' '.join(quoted_toks))
                        in_quotes, quoted_toks = False, []
                    else:
                        quoted_toks.append(t)
                else:
                    if t == "'": # entity start
                        in_quotes = True
                    else:
                        if len(t) > 1 and t.startswith('_'): # predicate remove prefix _
                            toks.append(t[1:])
                        else:
                            toks.append(t)
            return ''.join(toks)
        return [format_geo(lf) for lf in lf_list]

    def obtain_denotations(self, lf_list):
        tf = tempfile.NamedTemporaryFile('w+t', encoding='utf8', suffix='.dlog')
        tf_lines = ['_parse([query], %s).' % lf for lf in lf_list]
        for line in tf_lines:
            tf.write(line + '\n')
        tf.flush()
        msg = subprocess.check_output(['evaluator/geoquery', tf.name])
        msg = msg.decode('utf8')
        tf.close()

        def get_denotation(line):
            m = re.search('\{[^}]*\}', line)
            if m:
                return m.group(0)
            else:
                return line.strip()

        denotations = [
            get_denotation(line)
            for line in msg.split('\n')
            if line.startswith('        Example')
        ]
        return denotations

    def is_valid(self, ans_list):
        return list(map(lambda ans: 0.0 if 'FAILED' in ans or 'Join failed syntactically' in ans else 1.0, ans_list))
