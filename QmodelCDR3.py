#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:06:58 2019

@author: zacharysethna
"""

"""
Projection strings looks like:
Ln_AAp
"""

import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

#Load OLGA for seq generation
import olga.load_model as load_model
import olga.sequence_generation as seq_gen

class QmodelCDR3(object):
    
    def __init__(self, features = [], constant_features = [], data_seqs = [], gen_seqs = [], chain_type = 'humanTRB', load_model = None):
        self.features = np.array(features)
        self.constant_features =  constant_features
        self.model_params = np.zeros(len(self.features))
        self.data_seqs = []
        self.gen_seqs = []
        self.data_seq_features = []
        self.gen_seq_features = []
        self.data_marginals = np.array([])
        self.gen_marginals = np.array([])
        self.model_marginals = np.zeros(len(features))
        self.L1_converge_history = []
        default_chain_types = {'humanTRA': 'human_T_alpha', 'human_T_alpha': 'human_T_alpha', 'humanTRB': 'human_T_beta', 'human_T_beta': 'human_T_beta', 'humanIGH': 'human_B_heavy', 'human_B_heavy': 'human_B_heavy', 'mouseTRB': 'mouse_T_beta', 'mouse_T_beta': 'mouse_T_beta'}
        if chain_type not in default_chain_types.keys():
            print 'Unrecognized chain_type (not a default OLGA model). Please specify one of the following options: humanTRA, humanTRB, humanIGH, or mouseTRB.'
            return None
        self.chain_type = default_chain_types[chain_type]
        
        if load_model is not None:
            self.load_model(load_model)
        else:
            self.update_model(add_data_seqs = data_seqs, add_gen_seqs = gen_seqs)
            self.L1_converge_history = [sum(abs(self.data_marginals - self.model_marginals))]
        
        #self.floating_features = np.array([i for i in range(len(self.features)) if self.features[i] not in self.constant_features])
        self.floating_features = np.isin(self.features, self.constant_features, invert = True)
        
        self.max_iterations = 100
        self.step_size = 0.1 #step size
        self.converge_threshold = 1e-3
        self.l2_reg = None
        
        self.v = np.zeros(len(self.features))
        self.speed_reg = 0.95
        
        self.model_params_history = []
        
    def proj_feature_str_seq(self, feature_str, seq):
        """Checks if a sequence matches feature_str
        
        Parameters
        ----------
        feature_str : str
            Feature string to match. Looks like: Ln_AAp or gGENE
        seq : list
            CDR3 sequence and any associated genes
    
        Returns
        -------
        bool
            True if seq matches feature_str else False.
                
        """
        
        if feature_str[0] == 'v' or feature_str[0] == 'j': #check if there is a gene match
            try:
                return any([[int(x) for x in feature_str[1:].split('-')] == [int(y) for y in gene.lower().split(feature_str[0])[-1].split('-')] for gene in seq[1:] if feature_str[0] in gene.lower()])
            except ValueError:
                print feature_str, seq
                ValueError
        else:
            seq = seq[0] #extract CDR3 seq -- now a str
        
        split_feature_str = feature_str.split('_')
        try:
            if len(seq) != int(split_feature_str[0][1:]): #check length, if matches, do nothing
                return 0
        except ValueError: #No length restriction
            pass
        
        for s_a_str in split_feature_str[1:]:
    #        c_aa = s_a_str[0]
    #        c_index = int(s_a_str[1:])
            try:
                if seq[int(s_a_str[1:])] != s_a_str[0]:
                    return False
            except IndexError: #out of sequence range
                return False
                
        return True
        
    def find_relevant_features(self, seq, features = None):
        """Checks if a sequence matches feature_str
        
        
        Parameters
        ----------
        seq : list
            CDR3 sequence and any associated genes
        features : str
            List of feature strings to match.
    
        Returns
        -------
        relevant_features : list
            Indices of features seq projects onto.
                
        """
        if features is None:
            features = self.features
        relevant_features = []
        for feature_index,feature_str in enumerate(features):
            if self.proj_feature_str_seq(feature_str, seq):
                relevant_features += [feature_index]
                
        return relevant_features
    
    def compute_seq_energy(self, seq_features = None, seq = None):
        """Computes the energy of a sequence according to the model.
        
        
        Parameters
        ----------
        seq_features : list
            Features indexes seq projects onto.
        seq : list
            CDR3 sequence and any associated genes
    
        Returns
        -------
        E : float
            Energy of seq according to the model.
                
        """
        if seq_features is not None:
            return np.sum(self.model_params[seq_features])
        elif seq is not None:
            return np.sum(self.model_params[self.find_relevant_features(seq)])
        return 0
    
    def compute_marginals(self, seq_features_all, use_flat_distribution = False):
        """Computes the marginals of model features over sequences.
        
        Computes marginals either with a flat distribution over the sequences
        or weighted by the model energies.
        
        Parameters
        ----------
        seq_features_all : list
            Indices of features seqs project onto.
        
        Returns
        -------
        marginals : ndarray
            Marginals of model features over seqs.
                
        """
        marginals = np.zeros(len(self.features)) #initialize marginals
        if len(seq_features_all) == 0:
            return marginals
        if use_flat_distribution:
            for seq_features in seq_features_all:
                marginals[seq_features] += 1.
            return marginals/len(seq_features_all)
        else:
            energies = np.zeros(len(seq_features_all))
            for i, seq_features in enumerate(seq_features_all):
                energies[i] = self.compute_seq_energy(seq_features = seq_features)
                marginals[seq_features] += np.exp(-energies[i])
            return marginals/np.sum(np.exp(-energies))
    
    def compute_cross_marginals(self, cross_features, seqs, use_flat_distribution = False):
        """Computes the marginals of model features over sequences.
        
        Computes marginals either with a flat distribution over the sequences
        or weighted by the model energies.
        
        Parameters
        ----------
        seq_features_all : list
            Indices of features seqs project onto.
        
        Returns
        -------
        marginals : ndarray
            Marginals of model features over seqs.
                
        """
        cross_marginals = np.zeros(len(cross_features)) #initialize marginals
        if len(seqs) == 0:
            return cross_marginals
        seq_cross_features_all = [self.find_relevant_features(seq, features = cross_features) for seq in seqs]
        if use_flat_distribution:
            for seq_cross_features in seq_cross_features_all:
                cross_marginals[seq_cross_features] += 1.
            return cross_marginals/len(seqs)
        else:
            energies = np.zeros(len(seqs))
            for i, seq_cross_features in enumerate(seq_cross_features_all):
                energies[i] = self.compute_seq_energy(seq = seqs[i])
                cross_marginals[seq_cross_features] += np.exp(-energies[i])
            return cross_marginals/np.sum(np.exp(-energies))
        
    def infer_selection(self, max_iterations = None, step_size = None, speed_reg = None, initialize = False, l2_reg = None, record_history = False):
        """Infer model parameters (model_params)
        
        
        Parameters
        ----------
        max_iterations : int or None
            Maximum number of iterations
        step_size : float or None
            Step size for gradient descent.
        intialize : bool
            Resets model before inferring model_params
    
        Attributes set
        --------------
        model_params : array
            Energies for each model feature
        model_marginals : array
            Marginals over the generated sequences, reweighted by the model, 
            for each model feature.
        L1_converge_history : list
            L1 distance between data_marginals and model_marginals at each
            iteration.
                
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations
        if step_size is not None:
            self.step_size = step_size
        if speed_reg is not None:
            self.speed_reg = speed_reg
        self.l2_reg = l2_reg
            
        if initialize:
            #self.update_model(remove_features=[feature for i, feature in enumerate(self.features) if self.data_marginals[i] == 0 and self.gen_marginals[i] == 0])
            #self.model_params = -np.log(self.data_marginals/self.gen_marginals)
            #self.model_params = -np.log([self.data_marginals[i]/x if x > 0 else 1. for i, x in enumerate(self.gen_marginals)])
            self.model_params = np.array([-np.log10(self.data_marginals[i]/x) if x > 0 and self.data_marginals[i] > 0 else 0. for i, x in enumerate(self.gen_marginals)])
            self.update_model(auto_update_marginals=True) #make sure all attributes are updated
            self.L1_converge_history = [sum(abs(self.data_marginals - self.model_marginals))]
            self.model_params_history = []
            self.v = np.zeros(len(self.features))
        for _ in range(self.max_iterations):
            
            self.v = self.speed_reg*self.v + self.data_marginals - self.model_marginals
            if self.l2_reg is not None:
                self.model_params[self.floating_features] = (1-self.step_size*self.l2_reg)*self.model_params[self.floating_features] - self.step_size * self.v[self.floating_features] #new parameters
            else:
                self.model_params[self.floating_features] = self.model_params[self.floating_features] - self.step_size * self.v[self.floating_features] #new parameters
            
            self.model_marginals = self.compute_marginals(self.gen_seq_features)
            self.L1_converge_history.append(sum(abs(self.data_marginals - self.model_marginals)))
            if record_history:
                self.model_params_history.append(self.model_params)
            if self.L1_converge_history[-1] < self.converge_threshold:
                break


        
    def update_model(self, add_data_seqs = [], add_gen_seqs = [], add_features = [], remove_features = [], add_constant_features = [], auto_update_marginals = False):
        """Adds or removes model features to self.features
        
        
        Parameters
        ----------
        add_data_seqs : list
            List of CDR3 sequences to add to data_seq pool.
        add_gen_seqs : list
            List of CDR3 sequences to add to data_seq pool.
        add_gen_seqs : list
            List of CDR3 sequences to add to data_seq pool.
        add_features : list
            List of feature_strs to add to self.features
        remove_featurese : list
            List of feature_strs and/or indices to remove from self.features
    
        Attributes set
        --------------
        features : list
            List of model features
        data_seq_features : list
            Features data_seqs have been projected onto.
        gen_seq_features : list
            Features gen_seqs have been projected onto.
        data_marginals : dict
            Marginals over the data sequences for each model feature.
        gen_marginals : dict
            Marginals over the generated sequences for each model feature.
        model_marginals : dict
            Marginals over the generated sequences, reweighted by the model, 
            for each model feature.
                
        """
        
        self.data_seqs += [[seq,'',''] if type(seq)==str else seq for seq in add_data_seqs] #add_data_seqs
        self.gen_seqs += [[seq,'',''] if type(seq)==str else seq for seq in add_gen_seqs] #add_gen_seqs
        self.constant_features += add_constant_features
        
        if len(remove_features) > 0:
            indices_to_keep = [i for i, feature_str in enumerate(self.features) if feature_str not in remove_features and i not in remove_features]
            self.features = self.features[indices_to_keep]
            self.model_params = self.model_params[indices_to_keep]
            #self.floating_features = np.array([i for i in range(len(self.features)) if self.features[i] not in self.constant_features])
            self.floating_features = np.isin(self.features, self.constant_features, invert = True)
        if len(add_features) > 0:
            self.features = np.append(self.features, add_features)
            self.model_params = np.append(self.model_params, np.zeros(len(add_features)))
            #self.floating_features = np.array([i for i in range(len(self.features)) if self.features[i] not in self.constant_features])
            self.floating_features = np.isin(self.features, self.constant_features, invert = True)
            
        if len(add_data_seqs + add_features + remove_features) > 0 or auto_update_marginals:
            self.data_seq_features = [self.find_relevant_features(seq) for seq in self.data_seqs]
            self.data_marginals = self.compute_marginals(self.data_seq_features, use_flat_distribution = True)
            #self.inv_I = np.linalg.inv(self.compute_data_fisher_info())
        
        if len(add_gen_seqs + add_features + remove_features) > 0 or auto_update_marginals:
            self.gen_seq_features = [self.find_relevant_features(seq) for seq in self.gen_seqs]
            self.gen_marginals = self.compute_marginals(self.gen_seq_features, use_flat_distribution = True)
            self.model_marginals = self.compute_marginals(self.gen_seq_features)
        
        
        
    def add_orig_min_Q_model_features(self, min_L = 4, max_L = 22, include_genes = True):
        """Generates a list of feature_strs to implement the orig min Q model
        
        
        Parameters
        ----------
        min_L : int
            Minimum length CDR3 sequence
        max_L : int
            Maximum length CDR3 sequence
                
        """
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        features = []
        L_features = ['L' + str(L) for L in range(min_L, max_L + 1)]
        features += L_features
        for L in range(min_L, max_L + 1):
            for i in range(L):
                for aa in amino_acids:
                    features.append('L' + str(L) + '_' + aa + str(i))
                    
        if include_genes:
            main_folder = os.path.join(os.path.dirname(load_model.__file__), 'default_models', self.chain_type)
        
            params_file_name = os.path.join(main_folder,'model_params.txt')
            V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
            J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')
            
            genomic_data = load_model.GenomicDataVDJ()
            genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
            
            features += list(set(['v' + gen[0].split('*')[0].split('V')[-1] for gen in genomic_data.genV]))
            features += list(set(['j' + gen[0].split('*')[0].split('J')[-1] for gen in genomic_data.genJ]))
            
        self.update_model(add_features=features, add_constant_features=L_features)
        
    def add_z_Q_model_features(self, max_depth = 25, max_L = 30, include_genes = True):
        """Generates a list of feature_strs to implement the new Q model
        
        
        Parameters
        ----------
        max_L : int
            Maximum depth into CDR3 sequence considered
                
        """
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        features = []
        L_features = ['L' + str(L) for L in range(1, max_L + 1)]
        features += L_features
        for aa in amino_acids:
            features += ['_' + aa + str(L) for L in range(max_depth)]
            features += ['_' + aa + str(L) for L in range(-max_depth, 0)]
            
        if include_genes:
            main_folder = os.path.join(os.path.dirname(load_model.__file__), 'default_models', self.chain_type)
        
            params_file_name = os.path.join(main_folder,'model_params.txt')
            V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
            J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')
            
            genomic_data = load_model.GenomicDataVDJ()
            genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
            
            features += list(set(['v' + gen[0].split('*')[0].split('V')[-1] for gen in genomic_data.genV]))
            features += list(set(['j' + gen[0].split('*')[0].split('J')[-1] for gen in genomic_data.genJ]))
            
        self.update_model(add_features=features, add_constant_features=L_features)
    
    def add_generated_seqs(self, num_gen_seqs = 0, reset_gen_seqs = True):
        """Generates synthetic sequences for gen_seqs
        
        Generates sequences from one of the default OLGA VDJ recombination 
        models ('human_T_beta', 'human_B_heavy', or 'mouse_T_beta'). These
        sequences can be added to either the training sequence set or the
        validation sequence set. Updates t_seqs, v_seqs, t_vecs, and v_vecs.
        
        Requires the OLGA package (pip install olga).
        
        Parameters
        ----------
        num_synth_seqs : int or float
            Number of synthetic sequences to generate and add to the specified
            sequence pool.
        model_type : str
            Specifies default OLGA model for sequence generation. Options are:
            'human_T_beta' (default), 'human_B_heavy', or 'mouse_T_beta'.
    
        Attributes set
        --------------
        gen_seqs : list
            Synthetic sequences drawn from V(D)J recomb model
        gen_seq_features : list
            Features gen_seqs have been projected onto.
                
        """
        
        #Load generative model
        main_folder = os.path.join(os.path.dirname(load_model.__file__), 'default_models', self.chain_type)
        
        params_file_name = os.path.join(main_folder,'model_params.txt')
        marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
        V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')
        
        genomic_data = load_model.GenomicDataVDJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        generative_model = load_model.GenerativeModelVDJ()
        generative_model.load_and_process_igor_model(marginals_file_name)        
        sg_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)
        
        #Generate sequences
        seqs = [[seq[1], genomic_data.genV[seq[2]][0].split('*')[0], genomic_data.genJ[seq[3]][0].split('*')[0]] for seq in [sg_model.gen_rnd_prod_CDR3() for _ in range(int(num_gen_seqs))]]
        
        if reset_gen_seqs: #reset gen_seqs if needed
            self.gen_seqs = []
        #Add to specified pool(s)
        self.update_model(add_gen_seqs = seqs)
        
    def plot_model_learning(self, save_name = None):
        """Plots L1 convergence curve and marginal scatter.
        
        Parameters
        ----------
        save_name : str or None
            File name to save output figure. If None (default) does not save.
                
        """
        min_for_plot = 1/(10.*np.power(10, np.ceil(np.log10(len(self.data_seqs)))))
        fig = plt.figure(figsize = (9, 4))
        fig.add_subplot(121)
        fig.subplots_adjust(left=0.1, bottom = 0.13, top = 0.91, right = 0.97, wspace = 0.3, hspace = 0.15)
        plt.loglog(range(1, len(self.L1_converge_history)+1), self.L1_converge_history, 'k', linewidth = 2)
        plt.xlabel('Iteration', fontsize = 13)
        plt.ylabel('L1 Distance', fontsize = 13)
        
        plt.legend(frameon = False, loc = 2)
        plt.title('L1 Distance convergence', fontsize = 15)
        
        fig.add_subplot(122)
        
#        plt.loglog([self.data_marginals[p] for p in r_features], [self.gen_marginals[p] for p in r_features], 'r.', alpha = 0.2, markersize=1, label = 'Raw marginals')
#        plt.loglog([self.data_marginals[p] for p in r_features], [self.model_marginals[p] for p in r_features], 'b.', alpha = 0.2, markersize=1, label = 'Model adjusted marginals')
        plt.loglog(self.data_marginals, self.gen_marginals, 'r.', alpha = 0.2, markersize=1)
        plt.loglog(self.data_marginals, self.model_marginals, 'b.', alpha = 0.2, markersize=1)
        plt.loglog([],[], 'r.', label = 'Raw marginals')
        plt.loglog([],[], 'b.', label = 'Model adjusted marginals')

        plt.loglog([min_for_plot, 2], [min_for_plot, 2], 'k--', linewidth = 0.5)
        plt.xlim([min_for_plot, 1])
        plt.ylim([min_for_plot, 1])
        
        plt.xlabel('Marginals over data', fontsize = 13)
        plt.ylabel('Marginals over generated sequences', fontsize = 13)
        plt.legend(loc = 2, fontsize = 10)
        plt.title('Marginal Scatter', fontsize = 15)
            
        if save_name is not None:
            fig.savefig(save_name)
            
    def save_model(self, save_dir, attributes_to_save = None):
        """Saves model (energy_dict) to save_file as a tsv.
        
        Parameters
        ----------
        save_dir : str
            Directory name to save model attributes to.
            
        """
        
        if attributes_to_save is None:
            attributes_to_save = ['model_params', 'data_seqs', 'gen_seqs', 'L1_converge_history']
        
        if os.path.isdir(save_dir):
            if not raw_input('The directory ' + save_dir + ' already exists. Overwrite existing model (y/n)? ').strip().lower() in ['y', 'yes']:
                print 'Exiting...'
                return None

        os.mkdir(save_dir)
        
        if 'data_seqs' in attributes_to_save:
            with open(os.path.join(save_dir, 'data_seqs.tsv'), 'w') as data_seqs_file:
                data_seqs_file.write('\n'.join(['\t'.join(seq) for seq in self.data_seqs]))
        
        if 'gen_seqs' in attributes_to_save:
            with open(os.path.join(save_dir, 'gen_seqs.tsv'), 'w') as gen_seqs_file:
                gen_seqs_file.write('\n'.join(['\t'.join(seq) for seq in self.gen_seqs]))
        
        if 'L1_converge_history' in attributes_to_save:
            with open(os.path.join(save_dir, 'L1_converge_history.tsv'), 'w') as L1_file:
                L1_file.write('\n'.join([str(x) for x in self.L1_converge_history]))
        
        if 'model_params' in attributes_to_save:
            model_file = open(os.path.join(save_dir, 'model.tsv'), 'w')
            model_file.write('Feature\tEnergy\n')
            for i, p in enumerate(self.features):
                model_file.write(p + '\t' + str(self.model_params[i]) + '\n')
            model_file.close()
        
        return None
    
    def load_model(self, load_dir):
        """Loads model from directory.
        
        Parameters
        ----------
        load_dir : str
            Directory name to load model attributes from.
            
        """
        if not os.path.isdir(load_dir):
            print 'Directory for loading model does not exist (' + load_dir + ')'
            print 'Exiting...'
            return None
        
        if os.path.isfile(os.path.join(load_dir, 'model.tsv')):
            features = []
            model_params = []
            model_file = open(os.path.join(load_dir, 'model.tsv'), 'r')
            for i, line in enumerate(model_file):
                split_line = line.strip().split('\t')
                if i == 0: #skip header
                    continue
                try:
                    features.append(split_line[0])
                    model_params.append(float(split_line[1]))
                except:
                    pass
            model_file.close()
            self.features = np.array(features)
            self.model_params = np.array(model_params)
            
        else:
            print 'Cannot find model.tsv  --  no features or model parameters loaded.'
        
        if os.path.isfile(os.path.join(load_dir, 'data_seqs.tsv')):
            with open(os.path.join(load_dir, 'data_seqs.tsv'), 'r') as data_seqs_file:
                self.data_seqs = [seq.split('\t') for seq in data_seqs_file.read().strip().split('\n')]
        else:
            print 'Cannot find data_seqs.tsv  --  no data seqs loaded.'
            
        if os.path.isfile(os.path.join(load_dir, 'gen_seqs.tsv')):
            with open(os.path.join(load_dir, 'gen_seqs.tsv'), 'r') as gen_seqs_file:
                self.gen_seqs = [seq.split('\t') for seq in gen_seqs_file.read().strip().split('\n')]
        else:
            print 'Cannot find gen_seqs.tsv  --  no generated seqs loaded.'
        
        self.update_model(auto_update_marginals = True)
        
        if os.path.isfile(os.path.join(load_dir, 'L1_converge_history.tsv')):
            with open(os.path.join(load_dir, 'L1_converge_history.tsv'), 'r') as L1_file:
                self.L1_converge_history = [float(line.strip()) for line in L1_file if len(line.strip())>0]
        else:
            self.L1_converge_history = [sum([abs(self.data_marginals[p] - self.model_marginals[p]) for p in self.features])]
            print 'Cannot find L1_converge_history.tsv  --  no L1 convergence history loaded.'

        return None