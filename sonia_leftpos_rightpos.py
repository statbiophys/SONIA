#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: zacharysethna
"""


import os
from sonia import Sonia

class SoniaLeftposRightpos(Sonia):
	
	def __init__(self, data_seqs = [], gen_seqs = [], load_model = None, chain_type = 'humanTRB', max_depth = 25, max_L = 30, include_genes = True, seed = None,custom_pgen_model=None):
				
		Sonia.__init__(self, data_seqs=data_seqs, gen_seqs=gen_seqs, chain_type=chain_type, seed = seed)
		self.max_depth = max_depth
		self.max_L = max_L
		if load_model is not None:
			self.load_model(load_model)
		else:
			self.add_features(include_genes,custom_pgen_model)
	
	def add_features(self, include_genes = True,custom_pgen_model=None):
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
			
		if include_genes:
			import olga.load_model as olga_load_model
			if custom_pgen_model is None:
				main_folder = os.path.join(os.path.dirname(olga_load_model.__file__), 'default_models', self.chain_type)
			else:
				main_folder = self.custom_pgen_model        
			params_file_name = os.path.join(main_folder,'model_params.txt')
			V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
			J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')
			
			genomic_data = olga_load_model.GenomicDataVDJ()
			genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
			
			features += [[v, j] for v in set(['v' + genV[0].split('*')[0].split('V')[-1] for genV in genomic_data.genV]) for j in set(['j' + genJ[0].split('*')[0].split('J')[-1] for genJ in genomic_data.genJ])]
		
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
			v_genes = [gene for gene in seq[1:] if 'v' in gene.lower()]
			j_genes = [gene for gene in seq[1:] if 'j' in gene.lower()]
			#Allow for just the gene family match
			v_genes += [gene.split('-')[0] for gene in seq[1:] if 'v' in gene.lower()]
			j_genes += [gene.split('-')[0] for gene in seq[1:] if 'j' in gene.lower()]
			
			try:
				seq_feature_lsts += [['v' + '-'.join([str(int(y)) for y in gene.lower().split('v')[-1].split('-')])] for gene in v_genes]
				seq_feature_lsts += [['j' + '-'.join([str(int(y)) for y in gene.lower().split('j')[-1].split('-')])] for gene in j_genes]
				seq_feature_lsts += [['v' + '-'.join([str(int(y)) for y in v_gene.lower().split('v')[-1].split('-')]), 'j' + '-'.join([str(int(y)) for y in j_gene.lower().split('j')[-1].split('-')])] for v_gene in v_genes for j_gene in j_genes]
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
