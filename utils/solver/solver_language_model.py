# coding=utf8
import os, sys, time, gc
import numpy as np
import torch
from utils.solver.solver_base import Solver
from utils.batch import get_minibatch

class LMSolver(Solver):
    '''
        For traditional RNN-based Language Model
    '''
    def __init__(self, *args, **kargs):
        self.side = kargs.pop('side', 'question')
        super(LMSolver, self).__init__(*args, **kargs)
        self.best_result = {"losses": [], "iter": 0, "dev_ppl": float('inf'), "test_ppl": float('inf')}

    def decode(self, data_inputs, output_path, test_batchSize):
        data_index = np.arange(len(data_inputs))
        count, eval_loss, length_list = 0, [], []
        ########################### Evaluation Phase ############################
        self.model.eval()
        with open(output_path, 'w') as f:
            for j in range(0, len(data_index), test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, raw_inputs = get_minibatch(data_inputs, self.vocab, task='language_model',
                    data_index=data_index, index=j, batch_size=test_batchSize, device=self.device, side=self.side)
                length_list.extend((lens - 1).tolist())
                ########################## Calculate Sentence PPL #######################
                with torch.no_grad():
                    scores = self.model(inputs, lens) # bsize, seq_len, voc_size
                    batch_loss = self.loss_function(scores, inputs[:, 1:]).item()
                    eval_loss.append(batch_loss)
                    norm_log_prob = self.model.sent_logprobability(inputs, lens).cpu().tolist()

                ############################# Writing Result to File ###########################
                for idx in range(len(inputs)):
                    f.write('Utterance: ' + ' '.join(raw_inputs[idx]) + '\n')
                    f.write('NormLogProb: ' + str(norm_log_prob[idx]) + '\n')
                    current_ppl = np.exp(- norm_log_prob[idx])
                    f.write('PPL: ' + str(current_ppl) + '\n\n')

            ########################### Calculate Corpus PPL ###########################
            word_count = np.sum(length_list, axis=0)
            eval_loss = np.sum(eval_loss, axis=0)
            final_ppl = np.exp(eval_loss / word_count)
            f.write('Overall ppl: %.4f' % (final_ppl))
        return final_ppl

    def train_and_decode(self, train_inputs, dev_inputs, test_inputs, batchSize=16, test_batchSize=128, max_epoch=100):
        train_data_index = np.arange(len(train_inputs))
        nsentences = len(train_data_index)
        for i in range(max_epoch):
            ########################### Training Phase ############################
            start_time = time.time()
            np.random.shuffle(train_data_index)
            losses = []
            self.model.train()
            for j in range(0, nsentences, batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, _ = get_minibatch(train_inputs, self.vocab, task='language_model',
                    data_index=train_data_index, index=j, batch_size=batchSize, device=self.device, side=self.side)
                ############################ Forward Model ############################
                self.optimizer.zero_grad()
                batch_scores = self.model(inputs, lens)
                ############################ Loss Calculation #########################
                batch_loss = self.loss_function(batch_scores, inputs[:, 1:], lens - 1)
                losses.append(batch_loss.item())
                ########################### Backward and Optimize ######################
                batch_loss.backward()
                self.model.pad_embedding_grad_zero()
                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            epoch_loss = np.sum(losses, axis=0)
            self.best_result['losses'].append(epoch_loss)
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\t Loss of tgt : %.5f' \
                                % (i, time.time() - start_time, epoch_loss))
            gc.collect()
            torch.cuda.empty_cache()

            # whether evaluate later after training for some epochs
            if i < 10:
                continue

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_ppl = self.decode(dev_inputs, os.path.join(self.exp_path, 'valid.iter' + str(i)), test_batchSize)
            self.logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tppl : %.4f' % (i, time.time() - start_time, dev_ppl))
            start_time = time.time()
            test_ppl = self.decode(test_inputs, os.path.join(self.exp_path, 'test.iter' + str(i)), test_batchSize)
            self.logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tppl : %.4f' % (i, time.time() - start_time, test_ppl))

            ######################## Pick best result and save #####################
            if dev_ppl < self.best_result['dev_ppl']:
                self.model.save_model(os.path.join(self.exp_path, 'model.pkl'))
                self.best_result['iter'] = i
                self.best_result['dev_ppl'], self.best_result['test_ppl'] = dev_ppl, test_ppl
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid ppl : %.4f;\tBest Test ppl : %.4f' % (i, dev_ppl, test_ppl))

        ######################## Reload best model for later usage #####################
        self.logger.info('FINAL BEST RESULT: \tEpoch : %d\tBest Valid (ppl : %.4f)\tBest Test (ppl : %.4f) '
            % (self.best_result['iter'], self.best_result['dev_ppl'], self.best_result['test_ppl']))
        self.model.load_model(os.path.join(self.exp_path, 'model.pkl'))
