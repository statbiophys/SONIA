#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 30.04.2020
@author: Giulio Isacchini
"""

import multiprocessing as mp
import itertools
from tensorflow.keras.callbacks import Callback
import numpy as np

def add_random_error(nt, p):
    """ Take a nucleotide seq then simulate a sequencing
    error on it. Explicitely, each nucleotide has a probability p
    of being randomly modified. Adapted from Thomas Dupic.
    @ Arguments:
    * nt: amino-acid sequence
    * p: the error rate
    """
    rand = np.random.choice(["A", "T", "G", "C"], len(nt))
    return "".join([(a, r)[np.random.random() < p] for a, r in zip(nt, rand)])

def gene_to_num_str(gene_name, gene_type):
    """Strips excess gene name info to number string.

    Parameters
    ----------
    gene_name : str
        Gene or allele name
    gene_type : char
        Genomic cassette type. (i.e. V, D, or J)
    Returns
    -------
    num_str : str
        Reduced gene or allele name with leading zeros and excess
        characters removed.

    """
    # get rid of allele
    gene_name=gene_name.split('*')[0]
    num_str = gene_type.lower().join([g.lstrip('0') for g in gene_name.lower().split(gene_type.lower())[1:]])
    num_str = '-'.join([g.lstrip('0') for g in num_str.split('-')])
    return gene_type.lower() + num_str.replace('/', '')

def compute_pgen_expand(x):
    # compute pgen conditioned on gene usage
    return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj(x):
    # compute pgen unconditioned on gene usage
    return x[1].compute_aa_CDR3_pgen(x[0][0])

def partial_joint_marginals(args):
    # compute joint marginals on subset of seqs.
    features = args[0]
    Qs = args[1]
    marginals=args[2]
    l=int(np.sqrt(len(marginals)))
    Z=0
    for seq_features,Q in zip(features,Qs):
        for i,j in itertools.combinations(np.array(seq_features),2):
            if i>j:marginals[i,j] += Q
            else: marginals[j,i] += Q
        Z += Q
    return [marginals,Z]

class computeL1(Callback):
    # compute L1 distance at the end of each epoch of the inference.

    def __init__(self, sonia):
        self.data_marginals = sonia.data_marginals
        self.sonia=sonia
        self.len_features=len(sonia.features)
        self.gen_enc = self.sonia.X[self.sonia.Y.astype(np.bool)]
        self.encoded_data=self.sonia._encode_data(self.gen_enc)
        self.previous_loss=1e8
        
    def on_train_begin(self, logs={}):
        self.L1_history = []
        self.L1_history.append(np.sum(np.abs(self.return_model_marginals() - self.data_marginals)))
        self.weights_cpt=self.model.get_weights()
        print("Initial L1 dist: ", self.L1_history[-1])
        
    def return_model_marginals(self):
        marginals = np.zeros(self.len_features)
        Qs = np.exp(-self.model.predict(self.encoded_data)[:, 0])  
        for i in range(len(self.gen_enc)):
            marginals[self.gen_enc[i]] += Qs[i]
        return marginals / np.sum(Qs)
        
    def on_epoch_end(self, epoch, logs={}):
        curr_loss = logs.get('loss')
        curr_loss_val = logs.get('val_loss')
        if self.previous_loss > curr_loss_val:
            self.previous_loss=curr_loss_val
            self.weights_cpt=self.model.get_weights()
        self.L1_history.append(np.sum(np.abs(self.return_model_marginals() - self.data_marginals)))
        print("epoch = ", epoch, " loss = ", np.around(curr_loss, decimals=4) , " val_loss = ", 
              np.around(curr_loss_val, decimals=4), " L1 dist: ", np.around(self.L1_history[-1], decimals=4))
