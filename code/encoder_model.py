#!/usr/bin/python
# Author: Suzanna Sia

# Standard imports
#import random
import numpy as np
import pdb
import math
import os, sys

# Custom imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#torch.manual_seed(0)
#np.random.seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


class SENTEncoder(nn.Module):

    def __init__(self, z_dim=20, h_dim=300, encoder="", device="cpu"):
        super(SENTEncoder, self).__init__()
        self.device = device

        if encoder=="bert":
            enc_dim = 768
        elif encoder=="infersent":
            enc_dim=4096

        print("h_dim:", h_dim, encoder)

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fc1 = nn.Linear(enc_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        self.encoder_name = encoder

    def forward(self, last_hidden):
        # this takes an encoded representation from either BERT or Universal
        if self.encoder_name=="infersent":
            last_hidden = last_hidden.cuda()# to(self.device)

        h1 = F.relu(self.fc1(last_hidden))
        h2 = F.relu(self.fc2(h1))
        
        q_mu = self.fc_mu(h2)
        q_logvar = self.fc_logvar(h2)
        return q_mu, q_logvar


class RNNEncoder(nn.Module):
    def __init__(self, z_dim=20,
                        h_dim=100,
                        n_layers=1,
                        embedding_dim=300,
                        rnngate="lstm",
                        device='cpu'):
        super(RNNEncoder, self).__init__()

        self.n_layers = n_layers
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.rnn = getattr(nn, rnngate.upper())(embedding_dim, h_dim, n_layers,
                batch_first=True)

        self.rnngate = rnngate
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)


    def forward(self, embed, x_lengths, hidden=None):

        packed = pack_padded_sequence(embed, x_lengths, batch_first=True, enforce_sorted=False)
        if self.rnngate=="lstm":
            output_packed, (last_hidden, cell) = self.rnn(packed, hidden)
        else:
            output_packed, last_hidden = self.rnn(packed, hidden)
        #last_hidden = last_hidden.view(5, self.h_dim)
        h1 = F.relu(self.fc1(last_hidden))
        h2 = F.relu(self.fc2(h1))
        q_mu = self.fc_mu(h2)
        q_logvar = self.fc_logvar(h2)
 
#        q_mu = self.fc_mu(last_hidden)
#        q_logvar = self.fc_logvar(last_hidden)
        return q_mu, q_logvar

