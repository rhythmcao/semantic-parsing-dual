#coding=utf8
import sys, logging

def set_logger(exp_path, testing=False):
    logFormatter = logging.Formatter('%(asctime)s - %(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    if testing:
        fileHandler = logging.FileHandler('%s/log_test.txt' % (exp_path), mode='w')
    else:
        fileHandler = logging.FileHandler('%s/log_train.txt' % (exp_path), mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger