#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Giulio Isacchini

from __future__ import print_function, division,absolute_import

import numpy as np
import os
import multiprocessing as mp
import olga.load_model as olga_load_model
import olga.generation_probability as pgen
from sonia.utils import compute_pgen_expand,compute_pgen_expand_novj,partial_joint_marginals
import itertools
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
            try:
                if self.sonia_model.custom_pgen_model is None: main_folder = os.path.join(os.path.dirname(__file__), 'default_models', self.sonia_model.chain_type)
                else: main_folder=self.sonia_model.custom_pgen_model
            except:
                main_folder=os.path.join(os.path.dirname(__file__), 'default_models', self.sonia_model.chain_type)

            params_file_name = os.path.join(main_folder,'model_params.txt')
            marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
            V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
            J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

            if self.sonia_model.vj:
                self.genomic_data = olga_load_model.GenomicDataVJ()
                self.genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
                self.generative_model = olga_load_model.GenerativeModelVJ()
                self.generative_model.load_and_process_igor_model(marginals_file_name)
                self.pgen_model = pgen.GenerationProbabilityVJ(self.generative_model, self.genomic_data)
            else:
                self.genomic_data = olga_load_model.GenomicDataVDJ()
                self.genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
                self.generative_model = olga_load_model.GenerativeModelVDJ()
                self.generative_model.load_and_process_igor_model(marginals_file_name)
                self.pgen_model = pgen.GenerationProbabilityVDJ(self.generative_model, self.genomic_data)


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
        pgens=self.compute_all_pgens(seqs)/self.sonia_model.norm_productive # compute pgen
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

    def joint_marginals(self, features = None, seq_model_features = None, seqs = None, use_flat_distribution = False):
        '''Returns joint marginals P(i,j) with i and j features of sonia (l3, aA6, etc..), index of features attribute is preserved.
           Matrix is upper-triangular.

        Parameters
        ----------
        features: list
            custom feature list 
        seq_model_features: list
            encoded sequences
        seqs: list
            seqs to encode.
        use_flat_distribution: bool
            for data and generated seqs is True, for model is False (weights with Q)

        Returns
        -------
        joint_marginals: array
            matrix (i,j) of joint marginals

        '''

        if seq_model_features is None:  # if model features are not given
            if seqs is not None:  # if sequences are given, get model features from them
                seq_model_features = [self.sonia_model.find_seq_features(seq) for seq in seqs]
            else:   # if no sequences are given, sequences features is empty
                seq_model_features = []

        if len(seq_model_features) == 0:  # if no model features are given, return empty array
            return np.array([])

        if features is not None and seqs is not None:  # if a different set of features for the marginals is given
            seq_compute_features = [self.sonia_model.find_seq_features(seq, features = features) for seq in seqs]
        else:  # if no features or no sequences are provided, compute marginals using model features
            seq_compute_features = seq_model_features
            features = self.sonia_model.features
        l=len(features)
        two_points_marginals=np.zeros((l,l)) 
        n=len(seq_model_features)
        procs = mp.cpu_count()
        sizeSegment = int(n/procs)
        
        if not use_flat_distribution:
            energies = self.sonia_model.compute_energy(seq_model_features)
            Qs= np.exp(-energies)
        else:
            Qs=np.ones(len(seq_compute_features))
            
        # Create size segments list
        jobs = []
        for i in range(0, procs):
            jobs.append([seq_model_features[i*sizeSegment:(i+1)*sizeSegment],Qs[i*sizeSegment:(i+1)*sizeSegment],np.zeros((l,l))])
        p=mp.Pool(procs)
        pool = p.map(partial_joint_marginals, jobs)
        p.close()
        Z=np.array(pool)[:,1].sum()
        marg=np.array(pool)[:,0]
        for m in marg: two_points_marginals=two_points_marginals+m
        two_points_marginals= two_points_marginals/Z
        return two_points_marginals
    
    def joint_marginals_independent(self,marginals):
        '''Returns independent joint marginals P(i,j)=P(i)*P(j) with i and j features of sonia (l3, aA6, etc..), index of features attribute is preserved.
        Matrix is upper-triangular.

        Parameters
        ----------
        marginals: list
            marginals.

        Returns
        -------
        joint_marginals: array
            matrix (i,j) of joint marginals

        '''

        joint_marginals=np.zeros((len(marginals),len(marginals)))
        for i,j in itertools.combinations(np.arange(len(marginals)),2):
            if i>j:
                joint_marginals[i,j]=marginals[i]*marginals[j]
            else: 
                joint_marginals[j,i]=marginals[i]*marginals[j]
        return joint_marginals

    def compute_joint_marginals(self):
        '''Computes joint marginals for all.

        Attributes Set
        -------
        gen_marginals_two: array
            matrix (i,j) of joint marginals for pre-selection distribution
        data_marginals_two: array
            matrix (i,j) of joint marginals for data
        model_marginals_two: array
            matrix (i,j) of joint marginals for post-selection distribution
        gen_marginals_two_independent: array
            matrix (i,j) of independent joint marginals for pre-selection distribution
        data_marginals_two_independent: array
            matrix (i,j) of joint marginals for pre-selection distribution
        model_marginals_two_independent: array
            matrix (i,j) of joint marginals for pre-selection distribution
        '''

        self.gen_marginals_two = self.joint_marginals(seq_model_features = self.sonia_model.gen_seq_features, use_flat_distribution = True)
        self.data_marginals_two = self.joint_marginals(seq_model_features = self.sonia_model.data_seq_features, use_flat_distribution = True)
        self.model_marginals_two = self.joint_marginals(seq_model_features = self.sonia_model.gen_seq_features)
        self.gen_marginals_two_independent = self.joint_marginals_independent(self.sonia_model.gen_marginals)
        self.data_marginals_two_independent = self.joint_marginals_independent(self.sonia_model.data_marginals)
        self.model_marginals_two_independent = self.joint_marginals_independent(self.sonia_model.model_marginals)

    def compute_all_pgens(self,seqs):
        '''Compute Pgen of sequences using OLGA in parallel

        Parameters
        ----------
        seqs: list
            list of sequences to evaluate.

        Returns
        -------
        pgens: array
            generation probabilities of the sequences.

        '''

        final_models = [self.pgen_model for i in range(len(seqs))]    # every process needs to access this vector.
        pool = mp.Pool(processes=self.processes)

        if self.include_genes:
            f=pool.map(compute_pgen_expand, zip(seqs,final_models))
            pool.close()
            return np.array(f)
        else:
            f=pool.map(compute_pgen_expand_novj, zip(seqs,final_models))
            pool.close()
            return np.array(f)