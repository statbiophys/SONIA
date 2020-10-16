#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:12:15 2019

@author: administrator
"""

from __future__ import division,absolute_import
import numpy as np
import os
from sonia.sonia import Sonia
from sonia.utils import gene_to_num_str

#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass

class SoniaLengthPos(Sonia):

    def __init__(self, data_seqs = [], gen_seqs = [], chain_type = 'humanTRB',
                 load_dir = None, feature_file = None, data_seq_file = None, gen_seq_file = None, log_file = None,
                 min_L = 4, max_L = 30, include_indep_genes = False, include_joint_genes = True, min_energy_clip = -5, max_energy_clip = 10, seed = None,custom_pgen_model=None,l2_reg=0.,l1_reg=0.,vj=False):

        Sonia.__init__(self, data_seqs=data_seqs, gen_seqs=gen_seqs, chain_type=chain_type, min_energy_clip = min_energy_clip, max_energy_clip = max_energy_clip, seed = seed,l2_reg=l2_reg,l1_reg=0.,vj=vj)
        self.min_L = min_L
        self.max_L = max_L
        self.include_indep_genes=include_indep_genes
        self.include_joint_genes=include_joint_genes
        self.custom_pgen_model=custom_pgen_model
        if any([x is not None for x in [load_dir, feature_file]]):
            self.load_model(load_dir = load_dir, feature_file = feature_file, data_seq_file = data_seq_file, gen_seq_file = gen_seq_file, log_file = log_file)
        else:
            self.add_features(include_indep_genes = include_indep_genes, include_joint_genes = include_joint_genes, custom_pgen_model = custom_pgen_model)

    def add_features(self, include_indep_genes = False, include_joint_genes = True, custom_pgen_model=None):
        """Generates a list of feature_lsts for a length dependent L pos model.

        Parameters
        ----------
        include_genes : bool
            If true, features for gene selection are also generated. Currently
            joint V/J pairs used.

        custom_pgen_model: string
            path to folder of custom olga model.

        """

        features = []
        L_features = [['l' + str(L)] for L in range(self.min_L, self.max_L + 1)]
        features += L_features
        for L in range(self.min_L, self.max_L + 1):
            for i in range(L):
                for aa in self.amino_acids:
                     features.append(['l' + str(L), 'a' + aa + str(i)])

        if include_indep_genes or include_joint_genes:
            import olga.load_model as olga_load_model
            if custom_pgen_model is None:
                main_folder = os.path.join(os.path.dirname(__file__), 'default_models', self.chain_type)
            else:
                main_folder = custom_pgen_model
            params_file_name = os.path.join(main_folder,'model_params.txt')
            V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
            J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

            if self.vj: genomic_data = olga_load_model.GenomicDataVJ()
            else: genomic_data = olga_load_model.GenomicDataVDJ()
            genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)

            if include_indep_genes:
                features += [[v] for v in set([gene_to_num_str(genV[0],'V') for genV in genomic_data.genV])]
                features += [[j] for j in set([gene_to_num_str(genJ[0],'J') for genJ in genomic_data.genJ])]
            if include_joint_genes:
                features += [[v, j] for v in set([gene_to_num_str(genV[0],'V') for genV in genomic_data.genV]) for j in set([gene_to_num_str(genJ[0],'J') for genJ in genomic_data.genJ])]

        self.update_model(add_features=features)

    def find_seq_features(self, seq, features = None):
        """Finds which features match seq

        If no features are provided, the length dependent amino acid model
        features will be assumed.

        Parameters
        ----------
        seq : list
            CDR3 sequence and any associated genes
        features : ndarray
            Array of feature lists. Each list contains individual subfeatures which
            all must be satisfied.

        Returns
        -------
        seq_features : list
            Indices of features seq projects onto.

        """
        if features is None:
            seq_feature_lsts = [['l' + str(len(seq[0]))]]
            seq_feature_lsts += [['l' + str(len(seq[0])), 'a' + aa + str(i)] for i, aa in enumerate(seq[0])]
            v_genes = [gene.split('*')[0] for gene in seq[1:] if 'v' in gene.lower()]
            j_genes = [gene.split('*')[0] for gene in seq[1:] if 'j' in gene.lower()]
            #Allow for just the gene family match
            v_genes += [gene.split('*')[0].split('-')[0] for gene in seq[1:] if 'v' in gene.lower()]
            j_genes += [gene.split('*')[0].split('-')[0] for gene in seq[1:] if 'j' in gene.lower()]

            try:
                seq_feature_lsts += [[gene_to_num_str(gene,'V')] for gene in v_genes]
                seq_feature_lsts += [[gene_to_num_str(gene,'J')] for gene in j_genes]
                seq_feature_lsts += [[gene_to_num_str(v_gene,'V'), gene_to_num_str(j_gene,'J')] for v_gene in v_genes for j_gene in j_genes]
            except ValueError:
                pass
            seq_features = list(set([self.feature_dict[tuple(f)] for f in seq_feature_lsts if tuple(f) in self.feature_dict]))
        else:
            seq_features = []
            for feature_index,feature_lst in enumerate(features):
                if self.seq_feature_proj(feature_lst, seq):
                    seq_features += [feature_index]

        return seq_features


    def get_energy_parameters(self, return_as_dict = False):
        """Extract energy terms from keras model and gauge.

        For the length dependent position model, the gauge is set so that at a
        given position, for a given length, we have:

        <q_i,aa;L>_gen|L = 1


        Parameters
        ----------
        min_L : int
            Minimum length CDR3 sequence, if not given taken from class attribute
        max_L : int
            Maximum length CDR3 sequence, if not given taken from class attribute

        """
        model_energy_parameters = self.model.get_weights()[0].flatten()
        for l in range(self.min_L, self.max_L + 1):
            if self.gen_marginals[self.feature_dict[('l' + str(l),)]]>0:
                for i in range(l):
                    G = sum([(self.gen_marginals[self.feature_dict[('l' + str(l), 'a' + aa + str(i))]]
                                    /self.gen_marginals[self.feature_dict[('l' + str(l),)]]) *
                                    np.exp(-model_energy_parameters[self.feature_dict[('l' + str(l), 'a' + aa + str(i))]])
                                    for aa in self.amino_acids])
                    for aa in self.amino_acids:
                        model_energy_parameters[self.feature_dict[('l' + str(l), 'a' + aa + str(i))]] += np.log(G)

        if return_as_dict:
            return {f: model_energy_parameters[self.feature_dict[f]] for f in self.feature_dict}
        else:
            return model_energy_parameters

    def compute_seq_energy_from_parameters(self,seqs = None, seqs_features = None):
        """Computes the energy of a list of sequences according to the model.

        This computes according to model parameters instead of the keras model.
        As a result, no clipping occurs.

        Parameters
        ----------
        seqs : list or None
            Sequence list for a single sequence or many.
        seqs_features : list or None
            list of sequence features for a single sequence or many.

        Returns
        -------
        E : float
            Energies of seqs according to the model.

        """
        if seqs_features is not None:
            try:
                if isinstance(seqs_features[0], int):
                    seqs_features = [seqs_features]
            except:
                return None
        elif seqs is not None:
            try:
                if isinstance(seqs[0], str):
                    seqs = [seqs]
            except:
                return None
            seqs_features = [self.find_seq_features(seq) for seq in seqs]
        else:
            return None
        feature_energies = self.get_energy_parameters()
        return np.array([np.sum(feature_energies[seq_features]) for seq_features in seqs_features])

    def save_model(self, save_dir, attributes_to_save = None,force=True):
        """Saves model parameters and sequences

        Parameters
        ----------
        save_dir : str
            Directory name to save model attributes to.

        attributes_to_save: list
            Names of attributes to save

        """

        if attributes_to_save is None:
            attributes_to_save = ['model', 'data_seqs', 'gen_seqs', 'log']

        if os.path.isdir(save_dir):
            if not force:
                if not input('The directory ' + save_dir + ' already exists. Overwrite existing model (y/n)? ').strip().lower() in ['y', 'yes']:
                    print('Exiting...')
                    return None
        else:
            os.mkdir(save_dir)

        if 'data_seqs' in attributes_to_save:
            with open(os.path.join(save_dir, 'data_seqs.tsv'), 'w') as data_seqs_file:
                data_seq_energies = self.compute_energy(self.data_seq_features)
                data_seqs_file.write('Sequence;Genes\tLog_10(Q)\tFeatures\n')
                data_seqs_file.write('\n'.join([';'.join(seq) + '\t' + str(-data_seq_energies[i]/np.log(10)) + '\t' + ';'.join([','.join(self.features[f]) for f in self.data_seq_features[i]]) for i, seq in enumerate(self.data_seqs)]))

        if 'gen_seqs' in attributes_to_save:
            with open(os.path.join(save_dir, 'gen_seqs.tsv'), 'w') as gen_seqs_file:
                gen_seq_energies = self.compute_energy(self.gen_seq_features)
                gen_seqs_file.write('Sequence;Genes\tLog_10(Q)\tFeatures\n')
                gen_seqs_file.write('\n'.join([';'.join(seq) + '\t' +  str(-gen_seq_energies[i]/np.log(10)) + '\t' + ';'.join([','.join(self.features[f]) for f in self.gen_seq_features[i]]) for i, seq in enumerate(self.gen_seqs)]))

        if 'log' in attributes_to_save: 
            with open(os.path.join(save_dir, 'log.txt'), 'w') as L1_file:
                L1_file.write('Z ='+str(self.Z)+'\n')
                L1_file.write('norm_productive ='+str(self.norm_productive)+'\n')
                L1_file.write('likelihood_train,likelihood_test\n')
                for i in range(len(self.likelihood_train)):
                    L1_file.write(str(self.likelihood_train[i])+','+str(self.likelihood_test[i])+'\n')

        if 'model' in attributes_to_save:
            model_energy_dict = self.get_energy_parameters(return_as_dict = True)
            with open(os.path.join(save_dir, 'features.tsv'), 'w') as feature_file:
                feature_file.write('Feature\tEnergy\n')
                feature_file.write('\n'.join([';'.join(f) + '\t' + str(model_energy_dict[tuple(f)]) for f in self.features]))
            #self.model.save(os.path.join(save_dir, 'model.h5'))

        return None

    def _load_features_and_model(self, feature_file, model_file = None, verbose = True):
        """Loads left+right features and sets up model.

        Ignores model_file.
        """

        if feature_file is None and verbose:
            print('No feature file provided --  no features loaded.')
        elif os.path.isfile(feature_file):
            with open(feature_file, 'r') as features_file:
                all_lines = features_file.read().strip().split('\n')[1:] #skip header
                features = np.array([l.split('\t')[0].split(';') for l in all_lines])
                feature_energies = np.array([float(l.split('\t')[-1]) for l in all_lines]).reshape((len(features), 1))
            self.features = features
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}
            self.update_model_structure(initialize=True)
            self.model.set_weights([feature_energies])
        elif verbose:
            print('Cannot find features file --  no features or model parameters loaded.')