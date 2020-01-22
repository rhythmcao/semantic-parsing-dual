#coding=utf8
''' 
    Construct exp directory according to hyper parameters 
'''
import os

EXP_PATH = 'exp'

def hyperparam_seq2seq(options):
    """Hyerparam string for semantic parsing and question generation."""
    task_path = 'task_%s' % (options.task)
    dataset_path = 'dataset_%s' % (options.dataset)
    ratio = 'labeled_%s' % (options.labeled)

    exp_name = 'copy__' if options.copy else ''
    exp_name += 'cell_%s__' % (options.cell)
    exp_name += 'emb_%s__' % (options.emb_size)
    exp_name += 'hidden_%s_x_%s__' % (options.hidden_dim, options.num_layers)
    exp_name += 'dropout_%s__' % (options.dropout)
    exp_name += 'reduce_%s__' % (options.reduction)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'l2_%s__' % (options.l2)
    exp_name += 'bsize_%s__' % (options.batchSize)
    exp_name += 'me_%s__' % (options.max_epoch)
    exp_name += 'beam_%s__' % (options.beam)
    exp_name += 'nbest_%s' % (options.n_best)
    return os.path.join(EXP_PATH, task_path, dataset_path, ratio, exp_name)

def hyperparam_lm(options):
    task = 'task_%s' % (options.task)
    dataset_path = 'dataset_%s' % (options.dataset)
    ratio = '%s__labeled_%s' % (options.side, options.labeled)

    exp_name = ''
    exp_name += 'cell_%s__' % (options.cell)
    exp_name += 'emb_%s__' % (options.emb_size)
    exp_name += 'hidden_%s_x_%s__' % (options.hidden_dim, options.num_layers)
    exp_name += 'dropout_%s__' % (options.dropout)
    exp_name += 'reduce_%s__' % (options.reduction)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'l2_%s__' % (options.l2)
    exp_name += 'bsize_%s__' % (options.batchSize)
    exp_name += 'me_%s' % (options.max_epoch)
    exp_name += '__decTied' if options.decoder_tied else ''
    return os.path.join(EXP_PATH, task, dataset_path, ratio, exp_name)

def hyperparam_pseudo_method(options):
    task = 'task_%s' % (options.task)
    dataset_path = 'dataset_%s' % (options.dataset)
    ratio = 'labeled_%s__unlabeled_%s' % (options.labeled, options.unlabeled)
    ratio += '__extra' if options.extra else ''

    exp_name = ''
    if 'copy__' in options.read_sp_model_path:
        exp_name += 'sp_attnptr__'
    else:
        exp_name += 'sp_attn__'
    if 'copy__' in options.read_qg_model_path:
        exp_name += 'qg_attnptr__'
    else:
        exp_name += 'qg_attn__' 
    exp_name += 'reduce_%s__' % (options.reduction)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'l2_%s__' % (options.l2)
    exp_name += 'bsize_%s__' % (options.batchSize)
    exp_name += 'me_%s__' % (options.max_epoch)
    exp_name += 'beam_%s__' % (options.beam)
    exp_name += 'nbest_%s__' % (options.n_best)
    exp_name += 'discount_%s__method_%s' % (options.discount, options.method)
    return os.path.join(EXP_PATH, task, dataset_path, ratio, exp_name)

def hyperparam_dual_learning(options):
    task = 'task_%s' % (options.task)
    dataset_path = 'dataset_%s' % (options.dataset)
    ratio = 'labeled_%s__unlabeled_%s' % (options.labeled, options.unlabeled)
    ratio += '__extra' if options.extra else ''

    exp_name = ''
    if 'copy__' in options.read_sp_model_path:
        exp_name += 'sp_attnptr__'
    else:
        exp_name += 'sp_attn__'
    if 'copy__' in options.read_qg_model_path:
        exp_name += 'qg_attnptr__'
    else:
        exp_name += 'qg_attn__' 
    exp_name += 'reduce_%s__' % (options.reduction)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'l2_%s__' % (options.l2)
    exp_name += 'bsize_%s__' % (options.batchSize)
    exp_name += 'me_%s__' % (options.max_epoch)
    exp_name += 'beam_%s__' % (options.beam)
    exp_name += 'nbest_%s__' % (options.n_best)
    exp_name += 'cycle_%s__' % (options.cycle)
    exp_name += 'sample_%s__alpha_%s__beta_%s' % (options.sample, options.alpha, options.beta)
    return os.path.join(EXP_PATH, task, dataset_path, ratio, exp_name)
