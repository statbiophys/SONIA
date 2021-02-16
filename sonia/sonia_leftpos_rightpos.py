#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: zacharysethna
"""
from __future__ import division,absolute_import
import os
import numpy as np
from sonia.sonia import Sonia
from sonia.utils import gene_to_num_str
#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass


class SoniaLeftposRightpos(Sonia):

    def __init__(self, data_seqs = [], gen_seqs = [], chain_type = 'humanTRB',
                 load_dir = None, feature_file = None, data_seq_file = None, gen_seq_file = None, log_file = None, load_seqs = True,
                 max_depth = 25, max_L = 30, include_indep_genes = False, include_joint_genes = True, min_energy_clip = -5, max_energy_clip = 10, seed = None,custom_pgen_model=None,l2_reg=0.,l1_reg=0.,vj=False):
        Sonia.__init__(self, data_seqs=data_seqs, gen_seqs=gen_seqs, chain_type=chain_type, min_energy_clip = min_energy_clip, max_energy_clip = max_energy_clip, seed = seed,l2_reg=l2_reg,l1_reg=l1_reg,vj=vj)
        self.max_depth = max_depth
        self.max_L = max_L
        self.include_indep_genes=include_indep_genes
        self.include_joint_genes=include_joint_genes
        self.custom_pgen_model=custom_pgen_model
        if any([x is not None for x in [load_dir, feature_file]]):
            self.load_model(load_dir = load_dir, feature_file = feature_file, data_seq_file = data_seq_file, gen_seq_file = gen_seq_file, log_file = log_file, load_seqs = load_seqs)
        else:
            self.add_features(include_indep_genes = include_indep_genes, include_joint_genes = include_joint_genes, custom_pgen_model = custom_pgen_model)

    def add_features(self, include_indep_genes = False, include_joint_genes = True, custom_pgen_model=None):
        """Generates a list of feature_lsts for L/R pos model.

        Parameters
        ----------
        include_genes : bool
            If true, features for gene selection are also generated. Currently
            joint V/J pairs used.

        custom_pgen_model: string
            path to folder of custom olga model.

        """
        features = []
        L_features = [['l' + str(L)] for L in range(1, self.max_L + 1)]
        features += L_features
        for aa in self.amino_acids:
            features += [['a' + aa + str(L)] for L in range(self.max_depth)]
            features += [['a' + aa + str(L)] for L in range(-self.max_depth, 0)]

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


        If no features are provided, the left/right indexing amino acid model
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
            seq_feature_lsts += [['a' + aa + str(i)] for i, aa in enumerate(seq[0])]
            seq_feature_lsts += [['a' + aa + str(-1-i)] for i, aa in enumerate(seq[0][::-1])]
            
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
        """Extract energy terms from keras model.

        """
        model_energy_parameters = self.model.get_weights()[0].flatten()

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

    def set_gauge(self):
        """
        sets gauge such as sum(q)_i =1 at each position of CDR3 (left and right).
        """

        model_energy_parameters = self.model.get_weights()[0].flatten()

        Gs_plus=[]
        for i in list(range(self.max_depth)):
            norm_p=sum([self.gen_marginals[self.feature_dict[( 'a' + aa + str(i),)]] for aa in self.amino_acids])
            if norm_p==0:            
                G=1
            else:
                G = sum([self.gen_marginals[self.feature_dict[( 'a' + aa + str(i),)]] /norm_p * 
                          np.exp(-model_energy_parameters[self.feature_dict[( 'a' + aa + str(i),)]])
                          for aa in self.amino_acids])
            Gs_plus.append(G)
            for aa in self.amino_acids: model_energy_parameters[self.feature_dict[( 'a' + aa + str(i),)]] += np.log(G)
        Gs_minus=[]     
        for i in list(range(-self.max_depth,0)):
            norm_p=sum([self.gen_marginals[self.feature_dict[( 'a' + aa + str(i),)]] for aa in self.amino_acids])
            if norm_p==0:            
                G=1
            else:
                G = sum([self.gen_marginals[self.feature_dict[( 'a' + aa + str(i),)]] /norm_p * 
                          np.exp(-model_energy_parameters[self.feature_dict[( 'a' + aa + str(i),)]])
                          for aa in self.amino_acids])
            Gs_minus.append(G)
            for aa in self.amino_acids: model_energy_parameters[self.feature_dict[( 'a' + aa + str(i),)]] += np.log(G)
        for i in range(1,self.max_L+1):
            for j in list(range(i,self.max_depth)):
                model_energy_parameters[self.feature_dict[( 'l' + str(i),)]] += np.log(Gs_plus[j])
            for j in list(range(-self.max_depth,-i)):
                model_energy_parameters[self.feature_dict[( 'l' + str(i),)]] += np.log(Gs_minus[j])
        delta_Z=np.sum([np.log(g) for g in Gs_plus])+np.sum([np.log(g) for g in Gs_minus])

        self.min_energy_clip=self.min_energy_clip+delta_Z
        self.max_energy_clip=self.max_energy_clip+delta_Z
        self.Z=self.Z*np.exp(-delta_Z)
        self.update_model_structure(initialize=True)
        self.model.set_weights([np.array([model_energy_parameters]).T])

    def save_model(self, save_dir, attributes_to_save = None,force=True):
        """Saves model parameters and sequences

        Parameters
        ----------
        save_dir : str
            Directory name to save model attributes to.

        attributes_to_save: list
            Names of attributes to save

        """
        import shutil
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
                data_seq_energies = self.compute_seq_energy_from_parameters(seqs_features = self.data_seq_features)
                data_seqs_file.write('Sequence;Genes\tLog_10(Q)\tFeatures\n')
                data_seqs_file.write('\n'.join([';'.join(seq) + '\t' + str(-data_seq_energies[i]/np.log(10)) + '\t' + ';'.join([','.join(self.features[f]) for f in self.data_seq_features[i]]) for i, seq in enumerate(self.data_seqs)]))

        if 'gen_seqs' in attributes_to_save:
            with open(os.path.join(save_dir, 'gen_seqs.tsv'), 'w') as gen_seqs_file:
                gen_seq_energies = self.compute_seq_energy_from_parameters(seqs_features = self.gen_seq_features)
                gen_seqs_file.write('Sequence;Genes\tLog_10(Q)\tFeatures\n')
                gen_seqs_file.write('\n'.join([';'.join(seq) + '\t' +  str(-gen_seq_energies[i]/np.log(10)) + '\t' + ';'.join([','.join(self.features[f]) for f in self.gen_seq_features[i]]) for i, seq in enumerate(self.gen_seqs)]))

        if 'log' in attributes_to_save: 
            with open(os.path.join(save_dir, 'log.txt'), 'w') as L1_file:
                L1_file.write('Z ='+str(self.Z)+'\n')
                L1_file.write('norm_productive ='+str(self.norm_productive)+'\n')
                L1_file.write('min_energy_clip ='+str(self.min_energy_clip)+'\n')
                L1_file.write('max_energy_clip ='+str(self.max_energy_clip)+'\n')
                L1_file.write('likelihood_train,likelihood_test\n')
                for i in range(len(self.likelihood_train)):
                    L1_file.write(str(self.likelihood_train[i])+','+str(self.likelihood_test[i])+'\n')

        if 'model' in attributes_to_save:
            model_energy_dict = self.get_energy_parameters(return_as_dict = True)
            with open(os.path.join(save_dir, 'features.tsv'), 'w') as feature_file:
                feature_file.write('Feature,Energy,Marginal_data,Marginal_model,Marginal_gen\n')
                for i in range(len(self.features)):
                    feature_file.write(';'.join(self.features[i])+','+ str(model_energy_dict[tuple(self.features[i])])+','+str(self.data_marginals[i])+','+str(self.model_marginals[i])+','+str(self.gen_marginals[i])+'\n')
            #self.model.save(os.path.join(save_dir, 'model.h5'))
                #save pgen model too.
        try:
            if self.custom_pgen_model is None: main_folder = os.path.join(os.path.dirname(__file__), 'default_models', self.chain_type)
            else: main_folder=self.custom_pgen_model
        except:
            main_folder = os.path.join(os.path.dirname(__file__), 'default_models', self.chain_type)
        shutil.copy2(os.path.join(main_folder,'model_params.txt'),save_dir)
        shutil.copy2(os.path.join(main_folder,'model_marginals.txt'),save_dir)
        shutil.copy2(os.path.join(main_folder,'V_gene_CDR3_anchors.csv'),save_dir)
        shutil.copy2(os.path.join(main_folder,'J_gene_CDR3_anchors.csv'),save_dir)

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
                splitted=[l.split(',') for l in all_lines]
                features = np.array([l[0].split(';') for l in splitted], dtype=object)
                feature_energies = np.array([float(l[1]) for l in splitted]).reshape((len(features), 1))
                data_marginals=[float(l[2])  for l in splitted]
                model_marginals=[float(l[3])  for l in splitted]
                gen_marginals=[float(l[4])  for l in splitted]
            self.features = features
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}
            self.data_marginals=data_marginals
            self.model_marginals=model_marginals
            self.gen_marginals=gen_marginals
            self.update_model_structure(initialize=True)
            self.model.set_weights([feature_energies])
        elif verbose:
            print('Cannot find features file --  no features or model parameters loaded.')
