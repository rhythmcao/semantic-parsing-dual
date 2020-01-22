#coding=utf8
import time, os, gc
from utils.solver.solver_base import Solver
from utils.example import Example
from utils.batch import get_minibatch
from utils.bleu import get_bleu_score
import numpy as np
import torch

class QGSolver(Solver):

    def __init__(self, *args, **kargs):
        super(QGSolver, self).__init__(*args, **kargs)
        self.best_result = { "losses": [], "iter": 0, "dev_bleu": 0., "test_bleu": 0. }

    def decode(self, data_inputs, output_path, test_batchSize, beam=5, n_best=1):
        data_index= np.arange(len(data_inputs))
        nsentences, candidate_list, references_list = len(data_index), [], []
        domain = Example.domain
        self.model.eval()
        with open(output_path, 'w') as of:
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, dec_inputs, _, _, copy_tokens, oov_list, (raw_inputs, raw_outputs) = get_minibatch(
                    data_inputs, self.vocab, task='question_generation', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device, copy=self.model.copy)
                ############################ Forward Model ############################
                with torch.no_grad():
                    results = self.model.decode_batch(inputs, lens, self.vocab.word2id, copy_tokens, beam_size=beam, n_best=n_best)
                    predictions = results["predictions"]
                    predictions = [each[0] for each in predictions]
                    predictions = domain.reverse(predictions, self.vocab.id2word, oov_list=oov_list)
                bleu_scores = domain.compare_question(predictions, raw_outputs)
                candidate_list.extend(predictions)
                references_list.extend([[ref] for ref in raw_outputs])
                ############################ Write result to file ############################
                for idx in range(len(raw_inputs)):
                    of.write("LogicalForm: " + ' '.join(raw_inputs[idx]) + '\n')
                    of.write("Target: " + ' '.join(raw_outputs[idx]) + '\n')
                    of.write("Pred0: " + ' '.join(predictions[idx]) + '\n')
                    of.write("Bleu: " + str(bleu_scores[idx]) + '\n\n')
            avg_bleu = get_bleu_score(candidate_list, references_list)
            of.write('Overall bleu is %.4f' % (avg_bleu))
        return avg_bleu

    def train_and_decode(self, train_dataset, dev_dataset, test_dataset, batchSize=16, test_batchSize=128,
            max_epoch=100, beam=5, n_best=1):
        train_data_index = np.arange(len(train_dataset))
        nsentences = len(train_data_index)
        for i in range(max_epoch):
            ########################### Training Phase ############################
            start_time = time.time()
            np.random.shuffle(train_data_index)
            losses = []
            self.model.train()
            for j in range(0, nsentences, batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, dec_inputs, dec_outputs, out_lens, copy_tokens, _, _ = get_minibatch(
                    train_dataset, self.vocab, task='question_generation', data_index=train_data_index,
                    index=j, batch_size=batchSize, device=self.device, copy=self.model.copy)
                ############################ Forward Model ############################
                self.model.zero_grad()
                batch_scores = self.model(inputs, lens, dec_inputs[:, :-1], copy_tokens)
                ############################ Loss Calculation #########################
                batch_loss = self.loss_function(batch_scores, dec_outputs[:, 1:], out_lens - 1)
                losses.append(batch_loss.item())
                ########################### Backward and Optimize ######################
                batch_loss.backward()
                self.model.pad_embedding_grad_zero()
                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            epoch_loss = np.sum(losses, axis=0)
            self.best_result['losses'].append(epoch_loss)
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\t Loss: %.5f' \
                                % (i, time.time() - start_time, epoch_loss))
            gc.collect()
            torch.cuda.empty_cache()

            if i < 10:
                continue

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_bleu = self.decode(dev_dataset, os.path.join(self.exp_path, 'valid.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Dev Evaluation:\tEpoch : %d\tTime : %.4fs\tBleu : %.4f' \
                                % (i, time.time() - start_time, dev_bleu))
            start_time = time.time()
            test_bleu = self.decode(test_dataset, os.path.join(self.exp_path, 'test.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Test Evaluation:\tEpoch : %d\tTime : %.4fs\tBleu : %.4f' \
                                % (i, time.time() - start_time, test_bleu))

            ######################## Pick best result on dev and save #####################
            if dev_bleu >= self.best_result['dev_bleu']:
                self.model.save_model(os.path.join(self.exp_path, 'model.pkl'))
                self.best_result['iter'] = i
                self.best_result['dev_bleu'], self.best_result['test_bleu'] = dev_bleu, test_bleu
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid Bleu : %.4f;\tBest Test Bleu : %.4f' % (i, dev_bleu, test_bleu))

        ######################## Reload best model for later usage #####################
        self.logger.info('FINAL BEST RESULT: \tEpoch : %d\tBest Valid (Bleu : %.4f)\tBest Test (Bleu : %.4f)'
                % (self.best_result['iter'], self.best_result['dev_bleu'], self.best_result['test_bleu']))
        self.model.load_model(os.path.join(self.exp_path, 'model.pkl'))
