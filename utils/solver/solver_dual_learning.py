#coding=utf8
import os, sys, time, gc
import numpy as np
import torch
from utils.constants import *
from utils.batch import get_minibatch
from utils.example import Example
from utils.bleu import get_bleu_score
from utils.solver.solver_base import Solver

class DualLearningSolver(Solver):
    '''
        For Dual Learning Solver
    '''
    def __init__(self, *args, **kargs):
        super(DualLearningSolver, self).__init__(*args, **kargs)
        self.best_result = {
            "iter_sp": 0, "dev_acc": 0., "test_acc": 0.,
            "iter_qg": 0, "dev_bleu": 0., "test_bleu": 0.
        }

    def decode(self, data_inputs, output_path, test_batchSize, beam=5, n_best=1):
        data_index = np.arange(len(data_inputs))
        nsentences = len(data_index)
        domain = Example.domain
        total, candidate_list, references_list = [], [], []
        ########################### Evaluation Phase ############################
        with open(output_path, 'w') as of:
            self.model.eval()
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, _, _, _, copy_tokens, oov_list, (raw_inputs, raw_outputs) = get_minibatch(
                    data_inputs, self.vocab['sp'], task='semantic_parsing', data_index=data_index, 
                    index=j, batch_size=test_batchSize, device=self.device['sp'], copy=self.model.sp_model.copy)
                ############################ Forward Model ############################
                with torch.no_grad():
                    results = self.model.decode_batch(inputs, lens, self.vocab['sp'].lf2id, copy_tokens, task='semantic_parsing', beam_size=beam, n_best=n_best)
                    predictions = results["predictions"]
                    predictions = [pred for each in predictions for pred in each]
                    predictions = domain.reverse(predictions, self.vocab['sp'].id2lf, oov_list=oov_list)
                accuracy = domain.compare_logical_form(predictions, raw_outputs, pick=True)
                total.extend(accuracy)
                ############################ Write result to file ############################
                for idx in range(len(raw_inputs)):
                    of.write("Utterance: " + ' '.join(raw_inputs[idx]) + '\n')
                    of.write("Target: " + ' '.join(raw_outputs[idx]) + '\n')
                    for i in range(n_best):
                        of.write("Pred" + str(i) + ": " + ' '.join(predictions[n_best * idx + i]) + '\n')
                    of.write("Correct: " + ("True" if accuracy[idx] == 1 else "False") + '\n\n')

            of.write('=' * 50 + '\n' + '=' * 50 + '\n\n')

            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, _, _, _, copy_tokens, oov_list, (raw_inputs, raw_outputs) = get_minibatch(
                    data_inputs, self.vocab['qg'], task='question_generation', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device['qg'], copy=self.model.qg_model.copy)
                ########################## Beam Search/Greed Decode #######################
                with torch.no_grad():
                    results = self.model.decode_batch(inputs, lens, self.vocab['qg'].word2id, copy_tokens, task='question_generation', beam_size=beam, n_best=n_best)
                    predictions = results["predictions"]
                    predictions = [each[0] for each in predictions]
                    predictions = domain.reverse(predictions, self.vocab['qg'].id2word, oov_list=oov_list)
                bleu_scores = domain.compare_question(predictions, raw_outputs)
                candidate_list.extend(predictions)
                references_list.extend([[ref] for ref in raw_outputs])
                ############################# Writing Result to File ###########################
                for idx in range(len(raw_inputs)):
                    of.write("LogicalForm: " + ' '.join(raw_inputs[idx]) + '\n')
                    of.write("Target: " + ' '.join(raw_outputs[idx]) + '\n')
                    of.write("Pred0: " + ' '.join(predictions[idx]) + '\n')
                    of.write("Bleu: " + str(bleu_scores[idx]) + '\n\n')
            ########################### Calculate accuracy ###########################
            acc = sum(total) / float(len(total))
            avg_bleu = get_bleu_score(candidate_list, references_list)
            of.write('Overall accuracy: %.4f | Overall bleu score: %.4f' % (acc, avg_bleu))
        return acc, avg_bleu

    def train_and_decode(self, labeled_train_dataset, q_unlabeled_train_dataset, lf_unlabeled_train_dataset, dev_dataset, test_dataset,
            batchSize, test_batchSize, cycle='sp+qg', max_epoch=100, beam=5, n_best=1):
        sp_unlabeled_train_index = np.arange(len(q_unlabeled_train_dataset))
        qg_unlabeled_train_index = np.arange(len(lf_unlabeled_train_dataset))
        labeled_train_index = np.arange(len(labeled_train_dataset))
        nsentences = max([len(q_unlabeled_train_dataset), len(lf_unlabeled_train_dataset), len(labeled_train_dataset)])
        for i in range(max_epoch):
            ########################### Training Phase ############################
            start_time = time.time()
            np.random.shuffle(sp_unlabeled_train_index)
            np.random.shuffle(qg_unlabeled_train_index)
            np.random.shuffle(labeled_train_index)
            losses = { 'sp': [], 'qg': [] }
            self.model.train()
            for j in range(0, nsentences, batchSize):
                self.model.zero_grad()

                ''' Cycle start from Semantic Parsing '''
                if 'sp' in cycle:
                    ###################### Obtain minibatch data ######################
                    inputs, lens, copy_tokens, oov_list, raw_in = get_minibatch(q_unlabeled_train_dataset, self.vocab['sp'], task='unlabeled_semantic_parsing',
                        data_index=sp_unlabeled_train_index, index=j, batch_size=batchSize, device=self.device['sp'], copy=self.model.sp_model.copy)
                    ######################## Forward Model ##########################
                    sp_loss, qg_loss = self.model(inputs, lens, copy_tokens, oov_list, raw_in, start_from='semantic_parsing')
                    losses['sp'].append(sp_loss.item())
                    losses['qg'].append(qg_loss.item())
                    sp_loss.backward()
                    qg_loss.backward()

                ''' Cycle start from Question Generation '''
                if 'qg' in cycle:
                    ###################### Obtain minibatch data ######################
                    inputs, lens, copy_tokens, oov_list, raw_in = get_minibatch(lf_unlabeled_train_dataset, self.vocab['qg'], task='unlabeled_question_generation',
                        data_index=qg_unlabeled_train_index, index=j, batch_size=batchSize, device=self.device['qg'], copy=self.model.qg_model.copy)
                    ########################### Forward Model ########################
                    sp_loss, qg_loss = self.model(inputs, lens, copy_tokens, oov_list, raw_in, start_from='question_generation')
                    losses['sp'].append(sp_loss.item())
                    losses['qg'].append(qg_loss.item())
                    sp_loss.backward()
                    qg_loss.backward()

                ''' Supervised Training '''
                if True:
                    ###################### Obtain minibatch data ######################
                    inputs, lens, dec_inputs, dec_outputs, out_lens, copy_tokens, _, _ = get_minibatch(
                        labeled_train_dataset, self.vocab['sp'], task='semantic_parsing',
                        data_index=labeled_train_index, index=j, batch_size=batchSize, device=self.device['sp'], copy=self.model.sp_model.copy)
                    ############################ Forward Model ############################
                    batch_scores = self.model.sp_model(inputs, lens, dec_inputs[:, :-1], copy_tokens)
                    batch_loss = self.loss_function['sp'](batch_scores, dec_outputs[:, 1:], out_lens - 1)
                    losses['sp'].append(batch_loss.item())
                    batch_loss.backward()

                    ###################### Obtain minibatch data ######################
                    inputs, lens, dec_inputs, dec_outputs, out_lens, copy_tokens, _, _ = get_minibatch(
                        labeled_train_dataset, self.vocab['qg'], task='question_generation',
                        data_index=labeled_train_index, index=j, batch_size=batchSize, device=self.device['qg'], copy=self.model.qg_model.copy)
                    ############################ Forward Model ############################
                    batch_scores = self.model.qg_model(inputs, lens, dec_inputs[:, :-1], copy_tokens)
                    batch_loss = self.loss_function['qg'](batch_scores, dec_outputs[:, 1:], out_lens - 1)
                    losses['qg'].append(batch_loss.item())
                    batch_loss.backward()

                self.model.pad_embedding_grad_zero()
                self.optimizer.step()
                gc.collect()
                torch.cuda.empty_cache()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            sp_loss, qg_loss = np.sum(losses['sp'], axis=0), np.sum(losses['qg'], axis=0)
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\tLoss(sp loss : %.4f ; qg loss : %.4f)' \
                % (i, time.time() - start_time, sp_loss, qg_loss))

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_acc, dev_bleu = self.decode(dev_dataset, os.path.join(self.exp_path, 'valid.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tSemantic Parsing (acc : %.4f)\tQuestion Generation (bleu : %.4f)' \
                                % (i, time.time() - start_time, dev_acc, dev_bleu))
            start_time = time.time()
            test_acc, test_bleu = self.decode(test_dataset, os.path.join(self.exp_path, 'test.iter' + str(i)),
                test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tSemantic Parsing (acc : %.4f)\tQuestion Generation (bleu : %.4f)' \
                                % (i, time.time() - start_time, test_acc, test_bleu))

            ######################## Pick best result and save #####################
            if dev_acc > self.best_result['dev_acc']:
                self.model.save_model(sp_save_dir=os.path.join(self.exp_path, 'sp_model.pkl'))
                self.best_result['iter_sp'] = i
                self.best_result['dev_acc'], self.best_result['test_acc'] = dev_acc, test_acc
                self.logger.info('NEW BEST Semantic Parsing:\tEpoch : %d\tBest Valid (acc : %.4f)\tBest Test (acc : %.4f)' \
                        % (i, dev_acc, test_acc))
            if dev_bleu >= self.best_result['dev_bleu']:
                self.model.save_model(qg_save_dir=os.path.join(self.exp_path, 'qg_model.pkl'))
                self.best_result['iter_qg'] = i
                self.best_result['dev_bleu'], self.best_result['test_bleu'] = dev_bleu, test_bleu
                self.logger.info('NEW BEST Question Generation:\tEpoch : %d\tBest Valid (bleu : %.4f)\tBest Test (bleu : %.4f)' \
                        % (i, dev_bleu, test_bleu))
            gc.collect()
            torch.cuda.empty_cache()

        ######################## Reload best model for later usage #####################
        self.logger.info('FINAL BEST Semantic Parsing RESULT: \tEpoch : %d\tBest Valid (acc : %.4f)\tBest Test (acc : %.4f)'
                % (self.best_result['iter_sp'], self.best_result['dev_acc'], self.best_result['test_acc']))
        self.logger.info('FINAL BEST Question Generation RESULT: \tEpoch : %d\tBest Valid (bleu : %.4f)\tBest Test (bleu : %.4f)'
                % (self.best_result['iter_qg'], self.best_result['dev_bleu'], self.best_result['test_bleu']))
        self.model.load_model(os.path.join(self.exp_path, 'sp_model.pkl'), os.path.join(self.exp_path, 'qg_model.pkl'))
