#!/usr/bin/python
# Author: Suzanna Sia

# Standard imports
#import random
import numpy as np
import pdb
import math
import os, sys

# argparser
#import argparse
#from distutils.util import str2bool
#argparser = argparser.ArgumentParser()
#argparser.add_argument('--x', type=float, default=0)

# Custom imports
import torch
from torch import nn
from torch.nn import functional as F

#torch.manual_seed(0)
#np.random.seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False



class CNN_z(nn.Module):
    """ Convolves a z-representations of a set of sentences into a single z representation """
    # This doesnt care about sentence interaction between OP and CO

    def __init__(self,latent_dim):
        super(CNN_z, self).__init__()
        filter_sizes = [1, 2, 3, 4, 5]
        num_filters = 36
        self.convs = nn.ModuleList([nn.Conv2d(
            in_channels = 1, 
            out_channels = num_filters, 
            kernel_size= [fs, latent_dim], padding=(fs -1, 0)) for fs in filter_sizes])
        
        self.fc = nn.Linear(num_filters * len(filter_sizes), latent_dim)
    
    def forward(self, zs):
        zs = zs.unsqueeze(1)
        h = [conv(zs) for conv in self.convs]
        h = [F.relu(k).squeeze(3) for k in h]
        h = [F.max_pool1d(k, k.size(2)).squeeze(2) for k in h]
        h = torch.cat(h, 1)
        z = self.fc(h)
        return z

        #self.fc = nn.Linear(num_filters * len(filter_sizes), nwords)

# fix this up
class FFN_z(nn.Module):
    # importance weight each sentence into a single z representation
    # this doesnt take into account OH when doing z summary

    def __init__(self, z_dim, h_dim):
        super(FFN_z, self).__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)

    def forward(self, zs):
        
        z_weights = F.relu(self.fc1(zs))
        z_weights = F.relu(self.fc2(z_weights))
        z_weights = z_weights.squeeze(2)
        z_weights = torch.exp(z_weights)/torch.exp(z_weights).sum()

        zs = zs.squeeze(0)
        z_avg = torch.mm(z_weights, zs)

        return z_avg

class FFN_pairwise_z(nn.Module):
    def __init__(self, z_dim, h_dim, device):
        super(FFN_pairwise_z, self).__init__()
        self.fc1 = nn.Linear(z_dim+z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.device = device

    def forward(self, OP_zs, CO_zs):
        # fill in the matrix
        # get from shape of OP_z and CO_z
        # this is akin to kronecker product except we are just taking the vector instead of
        # doing matrix multiplication

        # check whether this breaks the gradient flow
        score_matrix = torch.zeros((OP_zs.shape[1], CO_zs.shape[1])).cuda() #

        for i in range(OP_zs.shape[1]):
            for j in range(CO_zs.shape[1]):
                zz = torch.cat((OP_zs[:,i,:], CO_zs[:,j,:]), 1)
                zz = F.relu(self.fc1(zz))
                zz_score = F.relu(self.fc2(zz))
                score_matrix[i,j] = zz_score
        
        OP_weights = score_matrix.sum(dim=1)
        CO_weights = score_matrix.sum(dim=0)

        if OP_weights.sum() == 0:
            # default to uniform
            OP_weights = torch.FloatTensor(OP_weights.shape[0]).uniform_(0,1).cuda() #

        if CO_weights.sum() == 0:
            CO_weights = torch.FloatTensor(CO_weights.shape[0]).uniform_(0, 1).cuda() #

        
        OP_probs = (OP_weights)/(OP_weights).sum()
        CO_probs = (CO_weights)/(CO_weights).sum()

        OP_sum = torch.mm(OP_probs.unsqueeze(0), OP_zs.squeeze(0))
        CO_sum = torch.mm(CO_probs.unsqueeze(0), CO_zs.squeeze(0))
        #OP_sum = torch.zeros(1, self.z_dim).cuda() #
        #for k in range(OP_probs.shape[0]):
        #    OP_sum += OP_probs[k] * OP_zs[:,k,:]
        #CO_sum = torch.zeros(1, self.z_dim).cuda() #
        #for k in range(CO_probs.shape[0]):
        #    CO_sum += CO_probs[k] * CO_zs[:, k, :]
        return OP_sum, CO_sum

class RNNV_z(nn.Module):

    """ Sequence-wise the z representation into variational z"""

    def __init__(self, z_dim, variational=True, device="cpu"):
        super(RNNV_z, self).__init__()
        self.lstm = nn.LSTM(input_size=z_dim, num_layers=2, hidden_size=z_dim, batch_first=True,
                bidirectional=True)
        self.device=device
        self.hidden = torch.randn((1, 1, z_dim)).cuda() # to(self.device)
        self.bidirectional = True
        self.z_dim = z_dim
        
        if self.bidirectional:
            self.fc_mu = nn.Linear(z_dim*2, z_dim)
        else:
            self.fc_mu = nn.Linear(z_dim, z_dim)

        self.fc_logvar = nn.Linear(z_dim, z_dim)
        self.q_mu = None # for inspecting values
        self.q_logvar = None 
        self.variational=variational

    def get_params(self):
        return self.q_mu, self.q_logvar

    def forward(self, zs):
        hx = self.hidden
        while zs.dim()<3:
            zs = zs.unsqueeze(0)
        _, (hx, cx) = self.lstm(zs)

        if self.bidirectional:
            hxs = hx.view(2, 2, 1, self.z_dim)
            fbh = torch.cat((hxs[-1][0], hxs[-1][1]), 1)
            q_mu = self.fc_mu(fbh)

        else:
            q_mu = self.fc_mu(hx[-1])  # take last hidden output

        self.q_mu = q_mu

        if self.variational:
            q_logvar = self.fc_logvar(hx[-1])
            self.q_logvar = q_logvar
            return self.sample_z_reparam(q_mu, q_logvar)
        else:
            return q_mu


        #return self.sample_z_reparam(q_mu, q_logvar)

        #return q_mu, q_logvar

    def sample_z_reparam(self, q_mu, q_logvar):
        eps = torch.randn_like(q_logvar).cuda() #.to(self.device)
        z = q_mu + torch.exp(q_logvar*0.5) * eps
        return z.cuda() #.to(self.device)

#class RNN_z(nn.Module):
#    """ Sequence-wise the z representations into a single z"""

#    def __init__(self, z_dim, device="cpu"):

#        super(RNN_z, self).__init__()
#        self.lstm = nn.LSTM(input_size=z_dim, num_layers=2, hidden_size=z_dim,
#        batch_first=True, bidirectional=False)
#        self.device=device

#        self.hidden = torch.randn((1, 1, z_dim)).to(self.device)

#    def forward(self, zs):
        # make bidirectional?
#        hx = self.hidden
#        _, (hx, cx) = self.lstm(zs)
#        return hx[-1] #take last hidden output 

