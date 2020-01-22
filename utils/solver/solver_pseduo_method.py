#coding=utf8
import os, sys, time, gc
import numpy as np
import torch
from utils.constants import *
from utils.batch import get_minibatch
from utils.example import Example
from utils.bleu import get_bleu_score
from utils.solver.solver_base import Solver

class PseudoSolver(Solver):
    '''
        For Pseudo Method Solver
    '''
    def __init__(self, *args, **kargs):
        self.discount = kargs.pop('discount', 0.5)
        self.method = kargs.pop('method', 'constant')
        super(PseudoSolver, self).__init__(*args, **kargs)
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
            self.model['sp'].eval()
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, _, _, _, copy_tokens, oov_list, (raw_inputs, raw_outputs) = get_minibatch(
                    data_inputs, self.vocab['sp'], task='semantic_parsing', data_index=data_index, 
                    index=j, batch_size=test_batchSize, device=self.device['sp'], copy=self.model['sp'].copy)
                ############################ Forward Model ############################
                with torch.no_grad():
                    results = self.model['sp'].decode_batch(inputs, lens, self.vocab['sp'].lf2id, copy_tokens, beam_size=beam, n_best=n_best)
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

            self.model['qg'].eval()
            for j in range(0, nsentences, test_batchSize):
                ###################### Obtain minibatch data ######################
                inputs, lens, _, _, _, copy_tokens, oov_list, (raw_inputs, raw_outputs) = get_minibatch(
                    data_inputs, self.vocab['qg'], task='question_generation', data_index=data_index,
                    index=j, batch_size=test_batchSize, device=self.device['qg'], copy=self.model['qg'].copy)
                ########################## Beam Search/Greed Decode #######################
                with torch.no_grad():
                    results = self.model['qg'].decode_batch(inputs, lens, self.vocab['qg'].word2id, copy_tokens, beam_size=beam, n_best=n_best)
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

    def generate_pseudo_samples(self, conf, test_batchSize, beam, sp_samples=None, qg_samples=None):
        domain = Example.domain
        pseudo_samples = []
        if sp_samples:
            nsentences = len(sp_samples)
            data_index = np.arange(nsentences)
            self.model['sp'].eval()
            for j in range(0, nsentences, test_batchSize):
                inputs, lens, copy_tokens, oov_list, raw_inputs = get_minibatch(sp_samples, self.vocab['sp'], 
                    task='unlabeled_semantic_parsing', data_index=data_index, index=j, batch_size=test_batchSize, 
                    device=self.device['sp'], copy=self.model['sp'].copy)
                with torch.no_grad():
                    results = self.model['sp'].decode_batch(inputs, lens, self.vocab['sp'].lf2id, copy_tokens, beam_size=beam, n_best=beam)
                    predictions = results["predictions"]
                    predictions = [pred for each in predictions for pred in each]
                    predictions = domain.reverse(predictions, self.vocab['sp'].id2lf, oov_list=oov_list)
                    _, idxs = domain.pick_predictions(domain.obtain_denotations(domain.normalize(predictions)), n_best=beam)
                    predictions = [predictions[each] for each in idxs]
                pseudo_samples.extend([Example(' '.join(words), ' '.join(lfs), conf) for words, lfs in zip(raw_inputs, predictions)])
        if qg_samples:
            nsentences = len(qg_samples)
            data_index = np.arange(nsentences)
            self.model['qg'].eval()
            for j in range(0, nsentences, test_batchSize):
                inputs, lens, copy_tokens, oov_list, raw_inputs = get_minibatch(qg_samples, self.vocab['qg'],
                    task='unlabeled_question_generation', data_index=data_index, index=j, batch_size=test_batchSize,
                    device=self.device['qg'], copy=self.model['qg'].copy)
                with torch.no_grad():
                    results = self.model['qg'].decode_batch(inputs, lens, self.vocab['qg'].word2id, copy_tokens, beam_size=beam, n_best=beam)
                    predictions = results["predictions"]
                    predictions = [each[0] for each in predictions]
                    predictions = domain.reverse(predictions, self.vocab['qg'].id2word, oov_list=oov_list)
                pseudo_samples.extend([Example(' '.join(words), ' '.join(lfs), conf) for words, lfs in zip(predictions, raw_inputs)])
        return pseudo_samples

    def train_and_decode(self, labeled_train_dataset, q_unlabeled_train_dataset, lf_unlabeled_train_dataset, dev_dataset, test_dataset,
            batchSize, test_batchSize, max_epoch=100, beam=5, n_best=1):
        for i in range(max_epoch):
            ########################### Training Phase ############################
            start_time = time.time()
            if self.method == 'constant':
                conf = self.discount
            elif self.method == 'linear':
                conf = self.discount * (i + 1) / float(max_epoch)
            else:
                raise ValueError("[Error]: not recognized method !")
            pseudo_samples = self.generate_pseudo_samples(conf, test_batchSize, beam, q_unlabeled_train_dataset, lf_unlabeled_train_dataset)
            self.logger.info('Generate %d new pseudo samples with confidence %.4f in epoch %d' % (len(pseudo_samples), conf, i))
            cur_train_dataset = labeled_train_dataset + pseudo_samples
            nsentences = len(cur_train_dataset)
            train_index = np.arange(nsentences)
            np.random.shuffle(train_index)
            losses = { 'sp': [], 'qg': [] }
            self.model['sp'].train()
            self.model['qg'].train()
            for j in range(0, nsentences, batchSize):
                self.model['sp'].zero_grad()
                self.model['qg'].zero_grad()
                ###################### Obtain minibatch data ######################
                inputs, lens, dec_inputs, dec_outputs, out_lens, copy_tokens, conf = get_minibatch(
                    cur_train_dataset, self.vocab['sp'], task='pseudo_semantic_parsing',
                    data_index=train_index, index=j, batch_size=batchSize, device=self.device['sp'], copy=self.model['sp'].copy)
                ############################ Forward Model ############################
                batch_scores = self.model['sp'](inputs, lens, dec_inputs[:, :-1], copy_tokens)
                batch_loss = self.loss_function['sp'](batch_scores, dec_outputs[:, 1:], out_lens - 1, conf)
                losses['sp'].append(batch_loss.item())
                batch_loss.backward()

                ###################### Obtain minibatch data ######################
                inputs, lens, dec_inputs, dec_outputs, out_lens, copy_tokens, conf = get_minibatch(
                    cur_train_dataset, self.vocab['qg'], task='pseudo_question_generation',
                    data_index=train_index, index=j, batch_size=batchSize, device=self.device['qg'], copy=self.model['qg'].copy)
                ############################ Forward Model ############################
                batch_scores = self.model['qg'](inputs, lens, dec_inputs[:, :-1], copy_tokens)
                batch_loss = self.loss_function['qg'](batch_scores, dec_outputs[:, 1:], out_lens - 1, conf)
                losses['qg'].append(batch_loss.item())
                batch_loss.backward()

                self.model['sp'].pad_embedding_grad_zero()
                self.model['qg'].pad_embedding_grad_zero()
                self.optimizer.step()

            print('[learning] epoch %i >> %3.2f%%' % (i, 100), 'completed in %.2f (sec) <<' % (time.time() - start_time))
            sp_loss, qg_loss = np.sum(losses['sp']), np.sum(losses['qg'])
            self.logger.info('Training:\tEpoch : %d\tTime : %.4fs\tSemantic Parsing Loss (loss : %.4f) ; Question Generation Loss (loss : %.4f)' \
                    % (i, time.time() - start_time, sp_loss, qg_loss))

            gc.collect()
            torch.cuda.empty_cache()

            ########################### Evaluation Phase ############################
            start_time = time.time()
            dev_acc, dev_bleu = self.decode(dev_dataset, os.path.join(self.exp_path, 'valid.iter' + str(i)), test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Dev Evaluation:\tEpoch : %d\tTime : %.4fs\tSemantic Parsing (acc : %.4f)\tQuestion Generation (bleu : %.4f)' \
                                % (i, time.time() - start_time, dev_acc, dev_bleu))
            start_time = time.time()
            test_acc, test_bleu = self.decode(test_dataset, os.path.join(self.exp_path, 'test.iter' + str(i)), test_batchSize, beam=beam, n_best=n_best)
            self.logger.info('Test Evaluation:\tEpoch : %d\tTime : %.4fs\tSemantic Parsing (acc : %.4f)\tQuestion Generation (bleu : %.4f)' \
                                % (i, time.time() - start_time, test_acc, test_bleu))

            ######################## Pick best result and save #####################
            if dev_acc > self.best_result['dev_acc']:
                self.model['sp'].save_model(os.path.join(self.exp_path, 'sp_model.pkl'))
                self.best_result['iter_sp'] = i
                self.best_result['dev_acc'], self.best_result['test_acc'] = dev_acc, test_acc
                self.logger.info('NEW BEST Semantic Parsing:\tEpoch : %d\tBest Valid (acc : %.4f)\tBest Test (acc : %.4f)' \
                        % (i, dev_acc, test_acc))
            if dev_bleu >= self.best_result['dev_bleu']:
                self.model['qg'].save_model(os.path.join(self.exp_path, 'qg_model.pkl'))
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
        self.model['sp'].load_model(os.path.join(self.exp_path, 'sp_model.pkl'))
        self.model['qg'].load_model(os.path.join(self.exp_path, 'qg_model.pkl'))
