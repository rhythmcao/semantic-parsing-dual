#coding=utf8

class Solver():

    def __init__(self, model, vocab, loss_function, optimizer, exp_path, logger, device=None, **kargs):
        super(Solver, self).__init__()
        self.model = model
        self.vocab = vocab
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device

    def decode(self, data_inputs, output_path, test_batchSize, beam_size=5, n_best=1):
        raise NotImplementedError
    
    def train_and_decode(self, train_dataset, dev_dataset, test_dataset, batchSize=16, test_batchSize=128,
        max_epoch=100, beam_size=5, n_best=1):
        raise NotImplementedError
