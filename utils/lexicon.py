#coding=utf8
import random, os
import collections
import itertools

class Lexicon():
    """
        A Lexicon class used for entity mapping and reverse entity mapping (in pointer network)

        1. Entity mapping: mapping word phrase into entity, replaced after copy
        ['in', 'which', 'seasons', 'kob', 'bryant', 'made', '3', 'blocks'] => (kob bryant, en.player.kobe_bryant)
        ==> ['in', 'which', 'seasons', 'en.player.kobe_bryant', 'en.player.kobe_bryant', 'made', '3', 'blocks']
        For word phrases with multiple choices, entity that matches longer spans takes precedence (Longest Match First)

        2. Reverse entity mapping: transform input logical form entities into natural phrases, replaced before copy (actually before feeding into network)
        ['(', 'lambda', '$0', 'e', '(', 'and', '(', 'flight', '$0', ')', '(', 'during_day', '$0', 'late:pd', ')'] => (late:pd, late flight|late|night)
        ==> ['(', 'lambda', '$0', 'e', '(', 'and', '(', 'flight', '$0', ')', '(', 'during_day', '$0', 'late', 'flight', ')']
        Randomly select one natural phrase from multiple choices if question word is not available
        Otherwise, use exactly the longest word phrase in the question.
        Attention: Remember to add late, flight to logical form vocabulary
    """
    def __init__(self, dataset):
        super(Lexicon, self).__init__()
        self.phrase2entity = collections.OrderedDict()
        self.entity2phrase = collections.OrderedDict()
        self.seen_words = set()
        self._load_lexicon(dataset)

    def _load_lexicon(self, dataset):
        entries = []
        if dataset in ['atis', 'geo']:
            lexicon_path = os.path.join('data', dataset, dataset + '_lexicon.txt')
        else:
            lexicon_path = os.path.join('data', 'overnight', dataset + '_lexicon.txt')
        print('Start load lexicon from file %s ...' % (lexicon_path))
        with open(lexicon_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '': continue
                x, y = line.split(' :- NP : ')
                entries.append((x.strip(), y.strip()))
        self._add_entries(entries)

    def _add_entries(self, entries):
        for name, entity in entries:
            if entity not in self.entity2phrase:
                self.entity2phrase[entity] = [name]
            elif name not in self.entity2phrase[entity]:
                self.entity2phrase[entity].append(name)
            if name in self.phrase2entity:
                if self.phrase2entity[name] != entity: # we do not handle entity disambiguation
                    print('Collision detected: %s -> %s, %s' % (name, self.entries[name], entity))
                continue
            # Update self.seen_words
            for w in name.split(' '):
                self.seen_words.add(w)
            self.phrase2entity[name] = entity
        for entity in self.entity2phrase: # sorted according to length of noun phrases
            self.entity2phrase[entity] = sorted(self.entity2phrase[entity], key=lambda x: len(x), reverse=True)

    def entity_mapping(self, words):
        """
            @args:
                words: a list of words
            @return:
                mapped_words: a list of words, where words[i] is replaced with entity if available
        """
        entities = ['' for i in range(len(words))]
        index_pairs = sorted(list(itertools.combinations(range(len(words) + 1), 2)),
                                key=lambda x: x[0] - x[1])
        ret_entries = []

        for i, j in index_pairs:
            # Longest match first
            if any(x for x in entities[i: j]): continue
            span = ' '.join(words[i: j])
            if span in self.phrase2entity:
                entity = self.phrase2entity[span]
                for k in range(i, j):
                    entities[k] = entity
                ret_entries.append(((i, j), entity))
        mapped_words = [words[idx] if not item else item for idx, item in enumerate(entities)]
        return mapped_words

    def reverse_entity_mapping(self, tokens, words=None):
        """
            @args:
                tokens: a list of logical form tokens
                words: a list of words if available
            @return:
                mapped_tokens: a list of tokens, where tokens[i] is replaced with noun phrases if available,
                    prefer to use raw noun phrase in words if available
        """
        entities = ['' for each in tokens]
        words = ' '.join(words) if words and words != ['none'] else None
        for idx, tok in enumerate(tokens):
            if tok in self.entity2phrase:
                if words:
                    choices = self.entity2phrase[tok]
                    among_words = list(filter(lambda item: item in words, choices))
                    if len(among_words) > 0:
                        entities[idx] = among_words[0]
                    else:
                        entities[idx] = random.choice(self.entity2phrase[tok])
                else:
                    entities[idx] = random.choice(self.entity2phrase[tok])
        mapped_words = [tokens[idx] if not item else item for idx, item in enumerate(entities)]
        return ' '.join(mapped_words).split(' ')
