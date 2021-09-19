#!/usr/bin/env python3
# Author: Suzanna Sia

### Standard imports
#import random
import numpy as np
import inspect
import pdb
import math
#import os
import sys
#import argparse

### Third Party imports
import yaml
import pandas as pd
#from torch.utils.tensorboard import SummaryWriter
### Local/Custom imports

class StopLoss:
    def __init__(self, losses=[], min_epoch=10, falls_itr=5, trainvalid_margin=0.4, logdir=""):
        self.epoch_scores = {}
        self.loss_names = losses
        self.min_epoch = min_epoch
        self.falls_itr = falls_itr
        self.trainvalid_margin = trainvalid_margin
        self.max_epoch = -1
        #self.writer = SummaryWriter(logdir)

        for k in losses:
            self.epoch_scores[k] = []

    def update(self, losses):
        for k, v in losses.items():
            self.epoch_scores[k].append(v)

    def report_max(self):
        # refactor
        assert self.max_epoch != -1
        results_d = {}
        results_d['train_epochs'] = self.max_epoch
        results_d['train_score'] = np.around(self.epoch_scores['train'][self.max_epoch], 3)
        results_d['valid_score'] = np.around(self.epoch_scores['valid'][self.max_epoch], 3)
        results_d['test_score'] = np.around(self.epoch_scores['test'][self.max_epoch], 3)

        if 'test2' not in self.epoch_scores:
            test_loss2 = 0
        else:
            test_loss2 = np.around(self.epoch_scores['test2'][self.max_epoch], 3)
        results_d['test_score2'] = test_loss2
        return results_d



    def check(self, epoch, stop=False):

        def _get_lossm(max_epoch):
            loss_m = ""
            for loss in self.loss_names:
                score = self.epoch_scores[loss][max_epoch]
                loss_m += "{}:{:.3f}, ".format(loss, score)
            return loss_m



        if len(self.epoch_scores[self.loss_names[0]]) >= self.min_epoch:
            valid_scores = self.epoch_scores['valid']
            message = ""

            if self._decrem(valid_scores):
                message = "Validation falls for 5 iterations at itr:{}".format(epoch)
                stop=True
                
            elif self._trainvalid_margin(epoch):
                message = "Train - Validation > {} at itr:{}".format(self.trainvalid_margin, epoch)
                stop=True

            print(message)

        if stop:
            max_valid_epoch = np.argmax(self.epoch_scores['valid'])
            max_train_epoch = np.argmax(self.epoch_scores['train'])
            #max_train = self.epoch_scores['train'][max_train_epoch]

            self.max_epoch = max_valid_epoch
            
            losses_t = _get_lossm(max_train_epoch)
            losses_v = _get_lossm(max_valid_epoch)
        
            print("max train @epoch {} {}".format(max_train_epoch, losses_t))
            print("max valid @epoch {} {}".format(max_valid_epoch, losses_v))

            sys.stdout.flush()

        return stop

    def _trainvalid_margin(self, ep):
        return (self.epoch_scores['train'][ep] - self.epoch_scores['valid'][ep]) > self.trainvalid_margin


    def _decrem(self, scores):
        # if score falls continuously for 5 epochs
        for i in range(1, self.falls_itr+1):
            if scores[-i] > scores[-i-1] + 0.01: # threshold 
                return False
        return True

    def write_tb(self, epoch=0, typ="", **kwargs):

        vals = {}
        for mode, metricD in kwargs.items():
            vals[mode] = {}

            for k in metricD.keys():
                val = np.around(np.mean(metricD[k]), 5)
                vals[mode][k] = val

        # reverse
        scalar_d = {}
        for k in metricD.keys():
            scalar_d[k] = {}
            for mode in vals.keys():
                scalar_d[k][mode] = vals[mode][k]
        
        #for k in scalar_d.keys():
            #self.writer.add_scalars(f'{typ}-{k}', scalar_d[k], epoch) 
            # scalar_d[k] will be a dictionary of train, test, split




        

def construct_lossD(fn, type_=0):

    lossD = {}
    weightsD = {}
    #DB.dp({"lossD":lossD, "weightsD":weightsD})

    with open(fn, 'r') as f:
        configs = yaml.safe_load(f)

    losses = configs['loss']

    for k in losses.keys():
        if type(type_)==list:
            lossD[k] = []
        else:
            lossD[k] = 0
        weightsD[k] = losses[k]

    #DB.dp({"lossD":lossD, "weightsD":weightsD})

    return lossD, weightsD

def total_loss(weightsD, lossD):

    loss = 0
    for k in lossD.keys():
        if weightsD[k]==0:
            pass
        # something is weird here, when we do 0 x loss it doesn't register as 0..
        else:
            loss += float(weightsD[k]) * lossD[k] #.item()

    return loss

def copy_loss(weightsD, lossD, lossD_epoch):
    # check if our losses are in the same scale
    
    #fnn = inspect.currentframe().f_code.co_name
    #DB.dp({"wd":weightsD, "lossD":lossD, "lossD_epoch":lossD_epoch})

    for k in lossD.keys():
        # we don't consider acc or roc as a loss for backprop, but we still want to report its
        # scores.
        if k == "acc" or k == "roc":
            val = lossD[k]
        else:
            val = weightsD[k] * lossD[k]

        try:
            lossD_epoch[k].append(val.item())
        except:
            lossD_epoch[k].append(val)

    #DB.dp({"wd":weightsD, "lossD":lossD, "lossD_epoch":lossD_epoch})

    return lossD_epoch

def reset_loss(lossD):
    #return construct_lossD(fn, type_=0)
    for loss in lossD.keys():
        lossD[loss] = 0
    return lossD

def print_loss(lossD, mode="", epoch=None):

    #DB.dp({'lossD':lossD})
    string = "\t"
    sorted_keys = sorted(lossD.keys())
    
    vals = []
    for k in sorted_keys:
        if len(lossD[k])>0:
            val = np.around(np.mean(lossD[k]), 5)
        else:
            val = 0

        loss_string = "{:.3f}".format(val)
        vals.append(loss_string)


    #DB.dp({'sortedKeys':sorted_keys, 'vals':vals})
    print(pd.DataFrame([vals], columns = sorted_keys))


def get_discr_loss(discrim_model, OP_z, EPS):
    fake_OP = discrim_model(OP_z)
    OP_z_r = Variable(torch.randn(1, args.z_dim) * 5.).to(device)
    real_OP = discrim_model(OP_z_r)
    discr_loss = -torch.mean(torch.log(real_OP + EPS) + torch.log(1 - fake_OP + EPS))
    gen_loss = -torch.mean(torch.log(fake_OP + EPS))

    return discr_loss, gen_loss
