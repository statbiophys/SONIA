#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Giulio Isacchini

from __future__ import print_function, division,absolute_import

import numpy as np
import os
import multiprocessing as mp
import olga.load_model as olga_load_model
import olga.generation_probability as pgen

class EvaluateModel(object):
    """Class used to evaluate sequences with the sonia model: Ppost=Q*Pgen


    Attributes
    ----------
    sonia_model: object
        Sonia model. Loaded previously, do not put the path.

    include_genes: bool
        Conditioning on gene usage for pgen/ppost evaluation. Default: True

    processes: int
        Number of processes to use to infer pgen. Default: all.

    custom_olga_model: object
        Optional: already loaded custom generation_probability olga model.

    Methods
    ----------

    evaluate_seqs(seqs=[])
        Returns Q, pgen and ppost of a list of sequences.

    evaluate_selection_factors(seqs=[])
        Returns normalised selection factor Q (Ppost=Q*Pgen) of a list of sequences (faster than evaluate_seqs because it does not compute pgen and ppost)

    """

    def __init__(self,sonia_model=None,include_genes=True,processes=None,custom_olga_model=None):

        if type(sonia_model)==str or sonia_model is None:
            print('ERROR: you need to pass a Sonia object')
            return

        self.sonia_model=sonia_model
        self.include_genes=include_genes
        if processes is None: self.processes = mp.cpu_count()
        else: self.processes = processes

        # define olga model
        if custom_olga_model is not None:
            self.pgen_model = custom_olga_model
        else:
            main_folder=os.path.join(os.path.dirname(__file__), 'default_models', self.sonia_model.chain_type)

            params_file_name = os.path.join(main_folder,'model_params.txt')
            marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
            V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
            J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

            if self.sonia_model.vj:
                genomic_data = olga_load_model.GenomicDataVJ()
                genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
                generative_model = olga_load_model.GenerativeModelVJ()
                generative_model.load_and_process_igor_model(marginals_file_name)
                self.pgen_model = pgen.GenerationProbabilityVJ(generative_model, genomic_data)
            else:
                genomic_data = olga_load_model.GenomicDataVDJ()
                genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
                generative_model = olga_load_model.GenerativeModelVDJ()
                generative_model.load_and_process_igor_model(marginals_file_name)
                self.pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)


    def evaluate_seqs(self,seqs=[]):
        '''Returns selection factors, pgen and pposts of sequences.

        Parameters
        ----------
        seqs: list
            list of sequences to evaluate

        Returns
        -------
        Q: array
            selection factor Q (of Ppost=Q*Pgen) of the sequences

        pgens: array
            pgen of the sequences

        pposts: array
            ppost of the sequences
        '''

        seq_features = [self.sonia_model.find_seq_features(seq) for seq in seqs] #find seq features
        energies =self.sonia_model.compute_energy(seq_features) # compute energies
        Q= np.exp(-energies)/self.sonia_model.Z # compute Q
        pgens=np.array(compute_all_pgens(seqs,self.pgen_model,self.processes,self.include_genes))/self.sonia_model.norm_productive # compute pgen
        pposts=pgens*Q # compute ppost

        return Q, pgens, pposts

    def evaluate_selection_factors(self,seqs=[]):
        '''Returns normalised selection factor Q (of Ppost=Q*Pgen) of list of sequences (faster than evaluate_seqs because it does not compute pgen and ppost)

        Parameters
        ----------
        seqs: list
            list of sequences to evaluate

        Returns
        -------
        Q: array
            selection factor Q (of Ppost=Q*Pgen) of the sequences

        '''

        seq_features = [self.sonia_model.find_seq_features(seq) for seq in seqs] #find seq features
        energies =self.sonia_model.compute_energy(seq_features) # compute energies

        return np.exp(-energies)/self.sonia_model.Z

# some parallel utils for pgen computation
def compute_pgen_expand(x):
    return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj(x):
    return x[1].compute_aa_CDR3_pgen(x[0][0])

def compute_all_pgens(seqs,model=None,processes=None,include_genes=True):
    '''Compute Pgen of sequences using OLGA in parallel

    Parameters
    ----------
    model: object
        olga model for evaluation of pgen.
    processes: int
        number of parallel processes (default all).
    include_genes: bool
        condition of v,j usage

    Returns
    -------
    pgens: array
        generation probabilities of the sequences.

    '''

    final_models = [model for i in range(len(seqs))]    # every process needs to access this vector.
    pool = mp.Pool(processes=processes)

    if include_genes:
        f=pool.map(compute_pgen_expand, zip(seqs,final_models))
        pool.close()
        return f
    else:
        f=pool.map(compute_pgen_expand_novj, zip(seqs,final_models))
        pool.close()
        return f
