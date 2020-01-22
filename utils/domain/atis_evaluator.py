#coding=utf8
import sys, os
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def merge_variables(v1, v2):
    keys = set(v1.keys()) | set(v2.keys())
    merged_v = {}
    for k in keys:
        if k in v1:
            value1 = v1[k] if type(v1[k]) == set else set([v1[k]])
        else:
            value1 = set()
        if k in v2:
            value2 = v2[k] if type(v2[k]) == set else set([v2[k]])
        else:
            value2 = set()
        merged_v[k] = value1 | value2
        merged_v[k] = merged_v[k] if len(merged_v[k]) > 1 else merged_v[k].pop()
    return merged_v

class ATISEvaluator():

    def __init__(self):
        """
            entity: set of types, {'bool', 'flight', 'ci', 'ap', ... }
            unary: dict of unary predicate, value of each key is a type set
                "oneway": {"flight"}
            binary: dict of binary predicate, value of each key is a type-tuple set
                "from": {("flight", "ci"), ("flight", "ap")}
        """
        self.entity, self.comparable, self.unary, self.binary = self._load_ontology('data/atis/atis_ontology.txt')
        self.vars = ["$0", "$1", "$2", "$3", "$4"] # type must be consistent
        self.key_word_func = {
            "lambda": self.check_arguments_of_lambda, "exists": self.check_arguments_of_exists,
            "and": self.check_arguments_of_and_or, "or": self.check_arguments_of_and_or,
            "max": self.check_arguments_of_max_min, "min": self.check_arguments_of_max_min,
            "argmax": self.check_arguments_of_argmax_argmin, "argmin": self.check_arguments_of_argmax_argmin,
            "sum": self.check_arguments_of_sum, "count": self.check_arguments_of_count,
            ">": self.check_arguments_of_compare, "<": self.check_arguments_of_compare,
            "=": self.check_arguments_of_equal, "the": self.check_arguments_of_the,
            "not": self.check_arguments_of_not
        }

    def _load_ontology(self, path):
        entity, comparable, unary, binary = set({"flight", "bool"}), set(), {}, {}
        with open(path, 'r') as infile:
            for line in infile:
                line = line.strip()
                if line == '': continue
                if line.startswith('entity'):
                    ent, flag = line.split('\t')[1:3]
                    entity.add(ent)
                    if flag == 'yes':
                        comparable.add(ent)
                elif line.startswith('unary'):
                    u = line.split('\t')[1]
                    if u not in unary:
                        unary[u] = set()
                    unary[u].add("flight")
                elif line.startswith("cat"):
                    u, t = line.split('\t')[1:]
                    if u not in unary:
                        unary[u] = set()
                    unary[u].add(t)
                elif line.startswith('binary'):
                    if len(line.split('\t')) != 4:
                        print(line)
                    t1, b, t2 = line.split('\t')[1:]
                    t1, t2 = t1[t1.index('type:') + len('type:'):], t2[t2.index('type:') + len('type:'):]
                    if b not in binary:
                        binary[b] = set()
                    binary[b].add((t1, t2))
                else:
                    print('[Warning]: unrecognized line while loading ontology %s' % (line))
        for each in unary:
            types = unary[each]
            for t in types:
                if t not in entity:
                    print('[Error]: args type %s of unary %s not recognized' % (t, each))
        for each in binary:
            types = binary[each]
            for (t1, t2) in types:
                if t1 not in entity:
                    print('[Error]: 1st args type %s of binary %s not recognized' % (t1, each))
                elif t2 not in entity:
                    print('[Error]: 2nd args type %s of binary %s not recognized' % (t2, each))
        return entity, comparable, unary, binary

    def eval(self, lf):
        """
            lf(list): lisp tree of lambda calculus logical form, e.g.
                ['lambda', '$0', 'e', ['flight', '$0']]
            If type consistent, return 1.0, else 0.0
        """
        try:
            t, variables = self.check_type_consistency(lf, {})
            return 1.0
        except Exception as e:
            return 0.0

    def check_type_consistency(self, lf, variables={}):
        if type(lf) != list: # entity
            t, variables = self.get_entity_type(lf, variables)
            return t, variables
        p = lf[0]
        if p in self.key_word_func:
            return self.key_word_func[p](lf, variables)
        elif p in self.unary or p in self.binary:
            return self.check_arguments_of_predicate(lf, variables)
        else:
            raise ValueError('[ERROR]: not recognized predicate %s' % (p))

    def check_arguments_of_predicate(self, lf, variables={}):
        p = lf[0]
        if p in self.unary and p not in self.binary:
            return self.check_arguments_of_unary(lf, variables)
        elif p not in self.unary and p in self.binary:
            return self.check_arguments_of_binary(lf, variables)
        elif p in self.unary and p in self.binary: # ambiguous predicate
            if len(lf) == 3:
                return self.check_arguments_of_binary(lf, variables)
            try:
                another_variables = copy.deepcopy(variables)
                u_t, u_variables = self.check_arguments_of_unary(lf, another_variables)
                u_t = set([u_t]) if type(u_t) != set else u_t
            except:
                u_t, u_variables = set(), {}
            try:
                b_t, b_variables = self.check_arguments_of_binary(lf, variables)
                b_t = set([b_t]) if type(b_t) != set else b_t
            except:
                b_t, b_variables = set(), {}
            t = u_t | b_t
            if len(t) == 0:
                raise ValueError('[ERROR]: type inconsistency of predicate %s' % (p))
            t = t if len(t) > 1 else t.pop()
            variables = merge_variables(u_variables, b_variables)
            return t, variables
        else:
            raise ValueError('[ERROR]: not recognized predicate %s' % (p))

    def check_arguments_of_unary(self, lf, variables={}):
        if len(lf) != 2:
            raise ValueError('[ERROR]: unary predicate %s only allow one argument' % (lf[0]))
        p, args = lf[0], lf[1]
        allowed_types = self.unary[p]
        t, variables = self.check_type_consistency(args, variables)
        if type(t) == set:
            t = t & allowed_types
            if len(t) == 0:
                raise ValueError('[ERROR]: type inconsistency of unary predicate %s and argument %s' % (p, args))
            else:
                if args in self.vars: # args must be vars ?
                    variables[args] = t if len(t) > 1 else t.pop()
        else:
            if t not in allowed_types:
                raise ValueError('[ERROR]: type inconsistency of unary predicate %s and argument %s' % (p, args))
        return 'bool', variables

    def check_arguments_of_binary(self, lf, variables={}):
        p, args = lf[0], lf[1:]
        if len(args) == 1:
            allowed_types = self.binary[p]
            t1, variables = self.check_type_consistency(args[0], variables)
            if type(t1) == set:
                t1 = t1 & set([i for i, _ in allowed_types if i in t1])
                allowed_types = set([(i, j) for i, j in allowed_types if i in t1])
            else:
                allowed_types = set([(i, j) for i, j in allowed_types if i == t1])
            if len(allowed_types) == 0:
                raise ValueError('[ERROR]: type inconsistency of binary predicate %s and first argument %s' % (p, args[0]))
            if type(t1) == set and args[0] in self.vars:
                variables[args[0]] = t1 if len(t1) > 1 else t1.pop()
            t2 = set([j for _, j in allowed_types])
            t2 = t2 if len(t2) > 1 else t2.pop()
            return t2, variables
        elif len(args) == 2:
            allowed_types = self.binary[p]
            t1, variables = self.check_type_consistency(args[0], variables)
            if type(t1) == set:
                t1 = t1 & set([i for i, _ in allowed_types if i in t1])
                allowed_types = set([(i, j) for i, j in allowed_types if i in t1])
            else:
                allowed_types = set([(i, j) for i, j in allowed_types if i == t1])
            if len(allowed_types) == 0:
                raise ValueError('[ERROR]: type inconsistency of binary predicate %s and first argument %s' % (p, args[0]))
            t2, variables = self.check_type_consistency(args[1], variables)
            if type(t2) == set:
                t2 = t2 & set([j for _, j in allowed_types if j in t2])
                allowed_types = set([(i, j) for i, j in allowed_types if j in t2])
            else:
                allowed_types = set([(i, j) for i, j in allowed_types if j == t2])
            if len(allowed_types) == 0:
                raise ValueError('[ERROR]: type inconsistency of binary predicate %s and second argument %s' % (p, args[1]))
            if type(t1) == set and args[0] in self.vars:
                variables[args[0]] = t1 if len(t1) > 1 else t1.pop()
            if type(t2) == set and args[1] in self.vars:
                variables[args[1]] = t2 if len(t2) > 1 else t2.pop()
            return 'bool', variables
        else:
            raise ValueError('[ERROR]: binary predicate %s only allow one or two arguments' % (p))

    def get_entity_type(self, ent, variables={}):
        if ent in self.vars:
            return variables[ent], variables
        for e in self.entity:
            if ent.endswith(':' + e):
                return e, variables
            elif ent.startswith(e):
                for i in range(6):
                    if ent == e + str(i):
                        return e, variables
        raise ValueError('[ERROR]: not recoginized entity %s' % (ent))

    def check_arguments_of_lambda(self, lf, variables={}):
        if len(lf) != 4:
            raise ValueError('[ERROR]: lambda function %s must have three arguments' % (lf))
        v = lf[1]
        if v not in self.vars:
            raise ValueError('[ERROR]: first argument of lambda function %s must be in the form $\{number\}' % (lf))
        if v not in variables:
            variables[v] = self.entity # can be any type
        e = lf[2]
        if e not in ["e", "i"]:
            raise ValueError('[ERROR]: second argument of lambda function %s must be e or i' % (lf))
        c = lf[3]
        t, variables = self.check_type_consistency(c, variables)
        if t != 'bool' and 'bool' not in t:
            raise ValueError('[ERROR]: third argument of lambda function %s must be bool constraint' % (lf))
        return 'bool', variables

    def check_arguments_of_exists(self, lf, variables={}):
        if len(lf) != 3:
            raise ValueError('[ERROR]: exists function %s must have two arguments' % (lf))
        v = lf[1]
        if v not in self.vars:
            raise ValueError('[ERROR]: first argument of exists function %s must be in the form $\{number\}' % (lf))
        if v not in variables:
            variables[v] = self.entity # can be any type
        c = lf[2]
        t, variables = self.check_type_consistency(c, variables)
        if t != 'bool' and 'bool' not in t:
            raise ValueError('[ERROR]: second argument of exists function %s must be bool constraint' % (lf))
        return 'bool', variables

    def check_arguments_of_and_or(self, lf, variables={}):
        if len(lf) < 2:
            raise ValueError('[ERROR]: and/or function %s at least have one constraints' % (lf))
        for idx, c in enumerate(lf[1:]):
            t, variables = self.check_type_consistency(c, variables)
            if t != 'bool' and 'bool' not in t:
                raise ValueError('[ERROR]: %s-th argument of and/or function %s must be bool constraint' % (idx, c))
        return 'bool', variables

    def check_arguments_of_max_min(self, lf, variables={}):
        if len(lf) != 3:
            raise ValueError('[ERROR]: max/min function %s must have two arguments' % (lf))
        v, c = lf[1], lf[2]
        if v not in self.vars:
            raise ValueError('[ERROR]: first argument of max/min function %s must be in the form $\{number\}' % (lf))
        if v not in variables:
            variables[v] = self.comparable # can be any comparable type
        t, variables = self.check_type_consistency(c, variables)
        if t != 'bool' and 'bool' not in t:
            raise ValueError('[ERROR]: second argument of max/min function %s must be bool constraint' % (lf))
        return variables[v], variables

    def check_arguments_of_argmax_argmin(self, lf, variables={}):
        if len(lf) != 4:
            raise ValueError('[ERROR]: argmax/argmin function %s must have three arguments' % (lf))
        v, c, s = lf[1], lf[2], lf[3]
        if v not in self.vars:
            raise ValueError('[ERROR]: first argument of argmax/argmin function %s must be in the form $\{number\}' % (lf))
        if v not in variables:
            variables[v] = self.entity # can be any type
        t, variables = self.check_type_consistency(c, variables)
        if t != 'bool' and 'bool' not in t:
            raise ValueError('[ERROR]: second argument of argmin/argmax function %s must be bool constraint' % (lf))
        t, variables = self.check_type_consistency(s, variables)
        if type(t) == set:
            t = t & self.comparable
            if len(t) == 0:
                raise ValueError('[ERROR]: third argument of argmin/argmax function %s must be comparable type' % (lf))
        else:
            if t not in self.comparable:
                raise ValueError('[ERROR]: third argument of argmin/argmax function %s must be comparable type' % (lf))
        return variables[v], variables

    def check_arguments_of_sum(self, lf, variables={}):
        if len(lf) != 4:
            raise ValueError('[ERROR]: sum function %s must have three arguments' % (lf))
        v, c, s = lf[1], lf[2], lf[3]
        if v not in self.vars:
            raise ValueError('[ERROR]: first argument of sum function %s must be in the form $\{number\}' % (lf))
        if v not in variables:
            variables[v] = self.entity # can be any type
        t, variables = self.check_type_consistency(c, variables)
        if t != 'bool' and 'bool' not in t:
            raise ValueError('[ERROR]: second argument of sum function %s must be bool constraint' % (lf))
        t, variables = self.check_type_consistency(s, variables)
        t = [t] if type(t) != set else t
        if 'i' not in t:
            raise ValueError('[ERROR]: third argument of sum function %s must be integer(i) type' % (lf))
        return 'i', variables

    def check_arguments_of_count(self, lf, variables={}):
        if len(lf) != 3:
            raise ValueError('[ERROR]: count function %s must have two arguments' % (lf))
        v, c = lf[1], lf[2]
        if v not in self.vars:
            raise ValueError('[ERROR]: first argument of count function %s must be in the form $\{number\}' % (lf))
        if v not in variables:
            variables[v] = self.entity # can be any type
        t, variables = self.check_type_consistency(c, variables)
        if t != 'bool' and 'bool' not in t:
            raise ValueError('[ERROR]: second argument of count function %s must be bool constraint' % (lf))
        return 'i', variables

    def check_arguments_of_compare(self, lf, variables={}):
        if len(lf) != 3:
            raise ValueError('[ERROR]: >/< function %s must have two arguments' % (lf))
        t1, variables = self.check_type_consistency(lf[1], variables)
        t2, variables = self.check_type_consistency(lf[2], variables)
        t1 = set([t1]) if type(t1) != set else t1
        t2 = set([t2]) if type(t2) != set else t2
        t = t1 & t2 & self.comparable
        if len(t) == 0:
            raise ValueError('[ERROR]: arguments of >/< function %s must be the same comparable type' % (lf))
        elif len(t) == 1:
            t = t.pop()
            if lf[1] in self.vars:
                variables[lf[1]] = t
            if lf[2] in self.vars:
                variables[lf[2]] = t
        return 'bool', variables

    def check_arguments_of_equal(self, lf, variables={}):
        if len(lf) != 3:
            raise ValueError('[ERROR]: = function %s must have two arguments' % (lf))
        t1, variables = self.check_type_consistency(lf[1], variables)
        t2, variables = self.check_type_consistency(lf[2], variables)
        t1 = set([t1]) if type(t1) != set else t1
        t2 = set([t2]) if type(t2) != set else t2
        t = t1 & t2
        if len(t) == 0:
            raise ValueError('[ERROR]: arguments of = function %s must be the same type' % (lf))
        elif len(t) == 1:
            t = t.pop()
        if lf[1] in self.vars:
            variables[lf[1]] = t
        if lf[2] in self.vars:
            variables[lf[2]] = t
        return 'bool', variables

    def check_arguments_of_not(self, lf, variables={}):
        if len(lf) != 2:
            raise ValueError('[ERROR]: not function %s only allow one argument' % (lf))
        t, variables = self.check_type_consistency(lf[1], variables)
        if t != 'bool' and 'bool' not in t:
            raise ValueError('[ERROR]: argument of not function must be bool type' % (lf))
        return 'bool', variables

    def check_arguments_of_the(self, lf, variables={}):
        if len(lf) != 3:
            raise ValueError('[ERROR]: the function %s must have two arguments' % (lf))
        v, c = lf[1], lf[2]
        if v not in self.vars:
            raise ValueError('[ERROR]: first argument of the function %s must be in the form $\{number\}' % (lf))
        if v not in variables:
            variables[v] = self.entity # can be any type
        t, variables = self.check_type_consistency(c, variables)
        if t != 'bool' and 'bool' not in t:
            raise ValueError('[ERROR]: second argument of the function %s must be bool constraint' % (lf))
        return variables[v], variables

if __name__ == '__main__':

    e = ATISEvaluator()
    join = set(e.binary.keys()) & set(e.unary.keys())
    print(join)
