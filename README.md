# Semantic Parsing with Dual Learning

Source code and data for ACL 2019 Long Paper [_Semantic Parsing with Dual Learning_](https://www.aclweb.org/anthology/P19-1007.pdf).

----

## Setup

* First, create the environment

        conda create -n sp python=3.6
        source activate sp
        pip3 install -r requirements.txt

* Second, pull all the dependencies from remote repository, including `evaluator`, `lib` and `glove6B word embeddings`.

        ./pull_dependency.sh

* Construct vocabulary for all datasets in advance under corresponding directory `data`, run

        python3 utils/statistics.py

----

## **Dataset**

----

Experiments are conducted on two semantic parsing dataset **ATIS** and **OVERNIGHT**, including traditional __train__, __dev__ and __test__ files, plus elaborated __lexicon__ files for *entity mapping* and *reverse entity mapping* techniques, and __extra__ files for synthesized unlabeled logical forms. An additional ontology file are created for dataset **ATIS** since there is no evaluator available.

----

### **ATIS**

Categories:

- *atis_train.tsv*: training dataset, 4433 samples.
- *atis_dev.tsv*: validation dataset, 491 samples.
- *atis_test.tsv*: test dataset, 448 samples.
- *atis_extra.tsv*: synthesized logical forms (Lambda Calculus), 3797 samples.
- *atis_lexicon.txt*: each line specifies a one-to-one mapping between a natural language noun phrase and its corresponding entity representation in knowledge base, such as pair `(first class, fist:cl)`.
- *atis_ontology.txt*: specify all the entity types, unary, and binary predicates used in the logical form.

**Attention**: Since there is no evaluator for this domain, we provide a simple type consistency checker for the target logical form (`utils/domain/atis_evaluator.py`). *atis_train.tsv*, *aits_dev.tsv* and *atis_test.tsv* are preprocessed version provided by [Dong and Lapata (2018)](https://www.aclweb.org/anthology/P16-1004.pdf), where natural language queries are lowercased and stemmed with NLTK, and entity mentions are replaced by numbered markers. For example:

    flight from ci0 to ci1	( lambda $0 e ( and ( flight $0 ) ( from $0 ci0 ) ( to $0 ci1 ) ) )

----

### **OVERNIGHT**

It contains eight sub-domains in total, namely *basketball*, *blocks*, *calendar*, *housing*, *publications*, *recipes*, *restaurants* and *socialnetwork*.

- *[domain]_train.tsv*: training and dev dataset. There is no isolate validation dataset in **OVERNIGHT**. We follow the traditional 80%/20% (train/dev) split in experiments.
- *[domain]_test.tsv*: test datset.
- *[domain]_extra.tsv*: synthesized logical forms (Lambda DCS). We revise the template rules in [SEMPRE](https://github.com/percyliang/sempre) to generate new instances.
- *[domain]_lexicon.txt*: each line specifies a one-to-one mapping between a natural language noun phrase and its corresponding entity representation in knowledge base, such as pair `(kobe bryant, en.player.kobe_bryant
)`.

**Attention**: There is also a evaluator program provided by [Jia and Liang (2016)](https://www.aclweb.org/anthology/P16-1002.pdf) in each domain to obtain denotations (`utils/domain/domain_overnight.py`). Each sample in *[domain]_train.tsv* and *[domain]_test.tsv* is of the form:

    what player did not play point guard	( call SW.listValue ( call SW.getProperty ( ( lambda s ( call SW.filter ( var s ) ( string position ) ( string ! = ) en.position.point_guard ) ) ( call SW.domain ( string player ) ) ) ( string player ) ) )

----

## Experiments

----

### Semantic Parsing (Supervised|Pretrain)

Refer to script in `run/run_semantic_parsing.sh`, for example

    ./run/run_semantic_parsing.sh dataset_name [attn|attnptr] labeled

`dataset_name` must be in choices `[atis, basketball, blocks, calendar, housing, publications, recipes, restaurants, socialnetwork]` and `labeled` denotes the ratio of labeled examples in training set we are going to use.

----

### Question Generation (Supervised|Pretrain)

The procedure is similar to that of Semantic Parsing since we use similar model architecture.

    ./run/run_question_generation.sh dataset_name [attn|attnptr] labeled

----

### Language Model (Unsupervised|Pretrain)

Language model is used to calculate the validity reward during the closed cycle.

    ./run/run_language_model.sh dataset_name [question|logical_form]

----

### Pseudo Method (Semi-supervised)

Use pretrained models of Semantic Parsing and Question Generation to generate pseudo samples. Then shuffle these pseudo samples with labeled samples together to train an improved Semantic Parsing and Question Generation Model.

    ./run/run_pseudo_method.sh dataset_name [attn|attnptr] labeled

**Attention:** in the script `run/run_pseudo_method.sh`, `read_sp_model_path` and `read_qg_model_path` are paths to the pretrained models(semantic parsing and question generation). `labeled` and `seed` should be kept the same for both the pretraining phases and pseudo method. By default, model type (attn/attnptr) is the same for both semantic parsing and question generation models.

----

### Dual Learning (Semi-supervised)

Use pretrained models of semantic parsing, question generation and language models to form two closed cycles with different starting points. Combine dual reinforcement learning algorithm and supervised training together. Running script:

    ./run/run_dual_learning.sh dataset_name [attn|attnptr] labeled

**Attention:** in the script `run/run_dual_learning.sh`, `read_sp_model_path`, `read_qg_model_path`, `read_qlm_path` and `read_lflm_path` are paths to the pretrained models(semantic parsing, question generation, question language model and logical form language model). `labeled` and `seed` should be kept the same for both the pretraining phases and dual learning framework. By default, model type (attn/attnptr) is the same for both semantic parsing and question generation models.
