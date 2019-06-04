# **Semantic Parsing Dataset**

----

This repository stores two semantic parsing dataset **ATIS** and **OVERNIGHT**, including traditional __train__, __dev__ and __test__ files, plus elaborated __lexicon__ files for *entity mapping* and *reverse entity mapping* techniques, and __extra__ files for synthesized unlabeled logical forms used in ACL 2019 Long Paper [*Semantic Parsing with Dual Learning*](url).

----

## **ATIS**

Categories:

- *train.tsv*: training dataset, 4433 samples.
- *dev.tsv*: validation dataset, 491 samples.
- *test.tsv*: test dataset, 448 samples.
- *extra.tsv*: synthesized logical forms (Lambda Calculus), 4592 samples.
- *lexicon.txt*: Each line specifies a one-to-one mapping between a natural language noun phrase and its corresponding entity representation in knowledge base, such as pair `(first class, fist:cl)`.

**Attention**: *train.tsv*, *dev.tsv* and *test.tsv* are preprocessed version provided by [Dong and Lapata (2018)](https://arxiv.org/pdf/1601.01280.pdf), where natural language queries are lowercased and stemmed with NLTK, and entity mentions are replaced by numbered markers. For example:

    flight from ci0 to ci1	( lambda $0 e ( and ( flight $0 ) ( from $0 ci0 ) ( to $0 ci1 ) ) )


----

## **OVERNIGHT**

It contains eight sub-domains in total, namely *basketball*, *blocks*, *calendar*, *housing*, *publications*, *recipes*, *restaurants* and *socialnetwork*.

- *[domain]_train.tsv*: training and dev dataset. There is no isolate validation dataset in **OVERNIGHT**. We recommend follow the traditional 80%/20% (train/dev) split in experiments.
- *[domain]_test.tsv*: test datset.
- *[domain]_extra.tsv*: synthesized logical forms (Lambda DCS). We revise the template rules in [SEMPRE](https://github.com/percyliang/sempre) to generate new instances.
- *[domain]_lexicon.txt*: Each line specifies a one-to-one mapping between a natural language noun phrase and its corresponding entity representation in knowledge base, such as pair `(kobe bryant, en.player.kobe_bryant
)`.

**Attention**: There is also a evaluator program for valid logical forms in each domain to obtain denotations. Each sample in *[domain]_train.tsv* and *[domain]_test.tsv* is of the form:

    what player did not play point guard	( call SW.listValue ( call SW.getProperty ( ( lambda s ( call SW.filter ( var s ) ( string position ) ( string ! = ) en.position.point_guard ) ) ( call SW.domain ( string player ) ) ) ( string player ) ) )
