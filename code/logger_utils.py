#!/usr/bin/python
# Author: Suzanna Sia

import logging
import os
import pdb

def log_file(args):

    fil = "{}/{}-{}-{}/z{}-h{}-zthresh{}/ssl{}-s{}-wd{}-hyp{}.{}".format(
                                                args.savedir,
                                                args.framework,
                                                str(args.encoder), 
                                                args.zsum,
                                                str(args.z_dim),
                                                str(args.h_dim),
                                                str(args.triplet_thresh),
                                                str(args.ss_recon_loss),
                                                str(args.scale_pzvar),
                                                str(args.word_dropout),
                                                str(args.hyp),
                                                str(args.seed)) 
                                            
    return fil

def get_all_logger(typ="all", args=None):
    
    if typ=="err":
        log_train = get_nn_logger(mode="train", args=args)
        log_valid = get_nn_logger(mode="valid", args=args)
        log_test = get_nn_logger(mode="test", args=args)

    else:
        log_train = get_sample_logger(mode="train", args=args)
        log_valid = get_sample_logger(mode="valid", args=args)
        log_test = get_sample_logger(mode="test", args=args)

    return log_train, log_valid, log_test

def get_nn_logger(mode="train", args=None):
    
    logger = logging.getLogger("rnn-{}".format(mode))
    logger.setLevel(logging.DEBUG)

    fil = log_file(args)
    fil = 'logs/zquad/{}/{}.err'.format(mode, fil)

    fol = os.path.dirname(fil)

    if not os.path.isdir(fol):
        os.makedirs(fol)
  

    fh = logging.FileHandler(fil)
    formatter = logging.Formatter('%(message)s')

    fh.setFormatter(formatter)

    logger.addHandler(fh)
    if mode=="train" or mode=="valid":
        logger.info("epoch\ttloss\tkldloss\tdeltaloss\tbceloss")

    return logger


def get_sample_logger(mode="train", args=None):
    logger1 = logging.getLogger("rnn-sample-reconstruct")
    logger1.setLevel(logging.DEBUG)

    fil = log_file(args)
    fil = 'logs/zquad/{}/{}.log'.format(mode, fil)

    fol = os.path.dirname(fil)
    if not os.path.isdir(fol):
        os.makedirs(fol)
 
    fh = logging.FileHandler(fil)
        
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger1.addHandler(fh)

    return logger1

def get_save_dir(args):

    fil = log_file(args)
    #fil = 'models/zquad/{}/models'.format(fil)
    fol = os.path.dirname(fil)
    if not os.path.isdir(fol):
        os.makedirs(fol)

    return fil
