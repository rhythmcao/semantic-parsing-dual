#coding=utf8
import time, os, gc
from utils.solver.solver_base import Solver
from utils.example import Example
from utils.batch import get_minibatch
import numpy as np
import torch

class SPSolver(Solver):

    def __init__(self, *args, **kargs):
        super(SPSolver, self).__init__(*args, **kargs)
        self.best_result = { "losses": [], "iter": 0, "dev_acc": 0., "test_acc": 0. }

    def decode(self, data_inputs, output_path, test_batchSize, beam=5, n_best=1):
        data_index= np.arange(len(data_inputs))
        nsentences, total = len(data_index), []
        domain = Example.domain
        self.model.eval()
        with open(output_path, 'w') as of:
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, dec_inputs, _, _, copy_tokens, oov_list, (raw_inputs, raw_outputs) = get_minibatch(
                    data_inputs, self.vocab, task='semantic_parsing', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device, copy=self.model.copy)
                ############################ Forward Model ############################
                with torch.no_grad():
                    results = self.model.decode_batch(inputs, lens, self.vocab.lf2id, copy_tokens, beam_size=beam, n_best=n_best)
                    predictions = results["predictions"]
                    predictions = [pred for each in predictions for pred in each]
                    predictions = domain.reverse(predictions, self.vocab.id2lf, oov_list=oov_list)
                accuracy = domain.compare_logical_form(predictions, raw_outputs, pick=True)
                total.extend(accuracy)
                ############################ Write result to file ############################
                for idx in range(len(raw_inputs)):
                    of.write("Utterance: " + ' '.join(raw_inputs[idx]) + '\n')
                    of.write("Target: " + ' '.join(raw_outputs[idx]) + '\n')
                    for i in range(n_best):
                        of.write("Pred" + str(i) + ": " + ' '.join(predictions[n_best * idx + i]) + '\n')
                    of.write("Correct: " + ("True" if accuracy[idx] == 1 else "False") + '\n\n')
            acc = sum(total) / float(len(total))
            of.write('Overall accuracy is %.4f' % (acc))
        return acc

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
                    train_dataset, self.vocab, task='semantic_parsing', data_index=train_data_index,
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
            dev_acc = self.decode(dev_dataset, os.path.join(self.exp_path, 'valid.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Dev Evaluation:\tEpoch : %d\tTime : %.4fs\tAcc : %.4f' \
                                % (i, time.time() - start_time, dev_acc))
            start_time = time.time()
            test_acc = self.decode(test_dataset, os.path.join(self.exp_path, 'test.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Test Evaluation:\tEpoch : %d\tTime : %.4fs\tAcc : %.4f' \
                                % (i, time.time() - start_time, test_acc))

            ######################## Pick best result on dev and save #####################
            if dev_acc >= self.best_result['dev_acc']:
                self.model.save_model(os.path.join(self.exp_path, 'model.pkl'))
                self.best_result['iter'] = i
                self.best_result['dev_acc'], self.best_result['test_acc'] = dev_acc, test_acc
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid Acc : %.4f;\tBest Test Acc : %.4f' % (i, dev_acc, test_acc))

        ######################## Reload best model for later usage #####################
        self.logger.info('FINAL BEST RESULT: \tEpoch : %d\tBest Valid (Acc : %.4f)\tBest Test (Acc : %.4f)'
                % (self.best_result['iter'], self.best_result['dev_acc'], self.best_result['test_acc']))
        self.model.load_model(os.path.join(self.exp_path, 'model.pkl'))
