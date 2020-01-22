#coding=utf8
import argparse, os, sys, time, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vocab import Vocab
from utils.example import Example, split_dataset
from utils.optimizer import set_optimizer
from utils.loss import set_loss_function
from utils.seed import set_random_seed
from utils.logger import set_logger
from utils.gpu import set_torch_device
from utils.constants import *
from utils.solver.solver_pseduo_method import PseudoSolver
from utils.hyperparam import hyperparam_pseudo_method
from models.construct_models import construct_model as model

############################### Arguments parsing and Preparations ##############################

def main(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='pseudo method for semantic parsing')
    parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
    parser.add_argument('--dataset', required=True, help='which dataset to experiment on')
    parser.add_argument('--read_model_path', help='Testing mode, load sp and qg model path')
    # model params
    parser.add_argument('--read_sp_model_path', required=True, help='pretrained sp model')
    parser.add_argument('--read_qg_model_path', required=True, help='pretrained qg model path')
    # pseudo training paras
    parser.add_argument('--reduction', choices=['sum', 'mean'], default='sum')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-5, help='weight decay (L2 penalty)')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--test_batchSize', type=int, default=128, help='input batch size in decoding')
    parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epochs to train for')
    # special paras
    parser.add_argument('--beam', default=5, type=int, help='used during decoding time')
    parser.add_argument('--n_best', default=1, type=int, help='used during decoding time')
    parser.add_argument('--labeled', type=float, default=1.0, help='ratio of labeled samples')
    parser.add_argument('--unlabeled', type=float, default=1.0, help='ratio of unlabeled samples')
    parser.add_argument('--method', choices=['constant', 'linear'], help='how to change confidence during training')
    parser.add_argument('--discount', type=float, default=1.0, help="final confidence for pseudo examples")
    parser.add_argument('--deviceId', type=int, nargs=2, default=[-1, -1], help='gpu indexes for slu and nlg models respectively, -1:cpu')
    parser.add_argument('--seed', type=int, default=999, help='set initial random seed')
    parser.add_argument('--extra', action='store_true', help='whether use synthesized logical forms')
    opt = parser.parse_args(args)

    # Some Arguments Check
    assert opt.labeled > 0. and opt.labeled < 1.0
    assert opt.unlabeled > 0. and opt.unlabeled <= 1.0
    return opt

opt = main()

####################### Output path, logger, device and random seed configuration #################

exp_path = opt.read_model_path if opt.testing else hyperparam_pseudo_method(opt)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

logger = set_logger(exp_path, testing=opt.testing)
logger.info("Parameters: " + str(json.dumps(vars(opt), indent=4)))
logger.info("Experiment path: %s" % (exp_path))
sp_device, qg_device = set_torch_device(opt.deviceId[0]), set_torch_device(opt.deviceId[1])
set_random_seed(opt.seed, device='cuda')

################################ Vocab and Data Reader ###########################

sp_copy, qg_copy = 'copy__' in opt.read_sp_model_path, 'copy__' in opt.read_qg_model_path
sp_vocab, qg_vocab = Vocab(opt.dataset, task='semantic_parsing', copy=sp_copy), Vocab(opt.dataset, task='question_generation', copy=qg_copy)
logger.info("Semantic Parsing model vocabulary ...")
logger.info("Vocab size for input natural language sentence is: %s" % (len(sp_vocab.word2id)))
logger.info("Vocab size for output logical form is: %s" % (len(sp_vocab.lf2id)))

logger.info("Question Generation model vocabulary ...")
logger.info("Vocab size for input logical form is: %s" % (len(qg_vocab.lf2id)))
logger.info("Vocab size for output natural language sentence is: %s" % (len(qg_vocab.word2id)))

logger.info("Read dataset starts at %s" % (time.asctime(time.localtime(time.time()))))
Example.set_domain(opt.dataset)
if not opt.testing:
    train_dataset, dev_dataset = Example.load_dataset(choice='train')
    labeled_train_dataset, unlabeled_train_dataset = split_dataset(train_dataset, opt.labeled)
    unlabeled_train_dataset, _ = split_dataset(unlabeled_train_dataset, opt.unlabeled)
    if opt.extra:
        q_unlabeled_train_dataset = unlabeled_train_dataset
        lf_unlabeled_train_dataset = unlabeled_train_dataset + Example.load_dataset(choice='extra')
    else:
        q_unlabeled_train_dataset, lf_unlabeled_train_dataset = unlabeled_train_dataset, unlabeled_train_dataset
    logger.info("Labeled/Unlabeled train dataset size is: %s and %s" % (len(labeled_train_dataset), len(lf_unlabeled_train_dataset)))
    logger.info("Dev dataset size is: %s" % (len(dev_dataset)))
test_dataset = Example.load_dataset(choice='test')
logger.info("Test dataset size is: %s" % (len(test_dataset)))

###################################### Model Construction ########################################

if not opt.testing:
    params = { "read_sp_model_path": opt.read_sp_model_path, "read_qg_model_path": opt.read_qg_model_path }
    json.dump(params, open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
else:
    params = json.load(open(os.path.join(exp_path, "params.json"), 'r'))
sp_params = json.load(open(os.path.join(params['read_sp_model_path'], 'params.json'), 'r'))
sp_model = model(**sp_params)
qg_params = json.load(open(os.path.join(params['read_qg_model_path'], 'params.json'), 'r'))
qg_model = model(**qg_params)
if not opt.testing:
    sp_model.load_model(os.path.join(params['read_sp_model_path'], 'model.pkl'))
    logger.info("Load Semantic Parsing model from path %s" % (params['read_sp_model_path']))
    qg_model.load_model(os.path.join(params['read_qg_model_path'], 'model.pkl'))
    logger.info("Load Question Generation model from path %s" % (params['read_qg_model_path']))
else:
    sp_model.load_model(os.path.join(exp_path, 'sp_model.pkl'))
    logger.info("Load Semantic Parsing model from path %s" % (exp_path))
    qg_model.load_model(os.path.join(exp_path, 'qg_model.pkl'))
    logger.info("Load Question Generation model from path %s" % (exp_path))
sp_model, qg_model = sp_model.to(sp_device), qg_model.to(qg_device)

loss_function = {'sp': {}, 'qg': {}}
loss_function['sp'] = set_loss_function(ignore_index=sp_vocab.lf2id[PAD], reduction=opt.reduction)
loss_function['qg'] = set_loss_function(ignore_index=qg_vocab.word2id[PAD], reduction=opt.reduction)
optimizer = set_optimizer(sp_model, qg_model, lr=opt.lr, l2=opt.l2, max_norm=opt.max_norm)

###################################### Training and Decoding #######################################

train_model = {'sp': sp_model, 'qg': qg_model}
vocab = {'sp': sp_vocab, 'qg': qg_vocab}
device = {'sp': sp_device, 'qg': qg_device}
solver = PseudoSolver(train_model, vocab, loss_function, optimizer, exp_path, logger, device=device,
    discount=opt.discount, method=opt.method)
if not opt.testing:
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    solver.train_and_decode(labeled_train_dataset, q_unlabeled_train_dataset, lf_unlabeled_train_dataset, dev_dataset, test_dataset,
        batchSize=opt.batchSize, test_batchSize=opt.test_batchSize,
        max_epoch=opt.max_epoch, beam=opt.beam, n_best=opt.n_best)
else:
    logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    acc, bleu = solver.decode(test_dataset, os.path.join(exp_path, 'test.eval'), opt.test_batchSize, beam=opt.beam, n_best=opt.n_best)
    logger.info('Evaluation cost: %.4fs\tSemantic Parsing (acc : %.4f)\tQuestion Generation (bleu: %.4f)' 
        % (time.time() - start_time, acc, bleu))
