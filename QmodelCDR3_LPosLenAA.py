#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:12:15 2019

@author: administrator
"""

import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

#Load OLGA for seq generation
import olga.load_model as load_model
from QmodelCDR3 import QmodelCDR3

class QmodelCDR3_LPosLenAA(QmodelCDR3):
    
    def add_features(self, min_L = 4, max_L = 22, include_genes = True):
        """Generates a list of feature_strs to implement the orig min Q model
        
        
        Parameters
        ----------
        min_L : int
            Minimum length CDR3 sequence
        max_L : int
            Maximum length CDR3 sequence
                
        """
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']#'ACDEFGHIKLMNPQRSTVWY'
        features = []
        L_features = ['L' + str(L) for L in range(min_L, max_L + 1)]
        features += L_features
        for L in range(min_L, max_L + 1):
            for i in range(L):
                for aa in self.amino_acids:
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
        
        self.feature_dict = {x:i for i,x in enumerate(self.features)}
        
    def set_guage(self):

        L_min = min([int(x.split('_')[0][1:]) for x in self.features])
        L_max = max([int(x.split('_')[0][1:]) for x in self.features])
        
        for l in range(L_min, L_max + 1):
            for i in range(l):
                G = sum([self.gen_marginals[self.feature_dict['L' + str(l) + '_' + aa + str(i)]]
                                /self.gen_marginals[self.feature_dict['L' + str(l)]] * 
                                np.exp(self.model_params[self.feature_dict['L' + str(l) + '_' + aa + str(i)]]) 
                                for aa in self.amino_acids])
                for aa in self.amino_acids:
                    self.model_params[self.feature_dict['L' + str(l) + '_' + aa + str(i)]] -= np.log(G)   
                    
    def plot_onepoint_values(self, onepoint , L_min, L_max, min_val, max_value, 
                             title = '', cmap = 'seismic', bad_color = 'black', aa_color = 'white', marginals = False):
        from mpl_toolkits.axes_grid1 import AxesGrid
        
        if marginals: #style for plotting marginals
            cmap = 'plasma'
            bad_color = 'white'
            aa_color = 'black'
            
        amino_acids_dict = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
                       'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro', 'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val'}
        
        fig = plt.figure(figsize=(12, 8))
    
        grid = AxesGrid(fig, 111,
                    nrows_ncols=(4, 5),
                    axes_pad=0.05,
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.1,
                    share_all = True
                    )
        
        current_cmap = matplotlib.cm.get_cmap(name = cmap)
        current_cmap.set_bad(color = bad_color)
        current_cmap.set_under(color = 'gray')
        
        for a,aa in enumerate(self.amino_acids):
        
            M = np.empty((L_max - L_min + 1, L_max))
            M[:] = np.nan
        
            for l in range(L_min, L_max + 1):
                for i in range(l):
                    M[l-L_min,i] = onepoint[self.feature_dict['L' + str(l) + '_' + aa + str(i)]]
            
            im = grid[a].imshow(M, cmap = current_cmap, vmin = min_val, vmax = max_value)
            grid[a].text(0.75,0.7,amino_acids_dict[aa],transform=grid[a].transAxes, color = aa_color, fontsize = 'large', fontweight = 'bold')
    
        grid.cbar_axes[0].colorbar(im)


        
        grid.axes_llc.set_xticks(range(0, L_max, 2))
        grid.axes_llc.set_xticklabels(range(1, L_max + 1, 2))
        grid.axes_llc.set_yticks(range(0, L_max - L_min + 1 ))
        grid.axes_llc.set_yticklabels(range(L_min, L_max + 1))
        
        fig.suptitle(title, fontsize=20.00)

    def plot_model_parameters(self):
        self.plot_onepoint_values(onepoint = np.exp(-self.model_params), L_min = 8, L_max = 16, min_val = 0, max_value = 2, title = 'model parameters q=exp(-E)')
        
    def norm_marginals(self, marg):        
        length_correction = np.array([(marg[self.feature_dict[f.split('_')[0]]] if ('_' in f) else 1.0) for f in self.features])
        length_correction[length_correction == 0] = 1 #avoid dividing by zero if there is no seq of this length
        return marg / length_correction
        

    def plot_marginals_length_corrected(self, L_min = 8, L_max = 16, log_scale = True):
        if log_scale:
            pc = 1e-10 #pseudo count to add to marginals to avoid log of zero
            self.plot_onepoint_values(onepoint = np.log(self.norm_marginals(self.data_marginals) + pc), L_min=L_min, L_max=L_max ,
                                      min_val = -8, max_value = 0, title = 'log(data marginals)', marginals = True)
            self.plot_onepoint_values(onepoint = np.log(self.norm_marginals(self.gen_marginals) + pc), L_min=L_min, L_max=L_max, 
                                      min_val = -8, max_value = 0, title = 'log(generated marginals)', marginals = True)
            self.plot_onepoint_values(onepoint = np.log(self.norm_marginals(self.model_marginals) + pc), L_min=L_min, L_max=L_max, 
                                      min_val = -8, max_value = 0, title = 'log(model marginals)', marginals = True)
        else:
            self.plot_onepoint_values(onepoint = self.norm_marginals(self.data_marginals), L_min=L_min, L_max=L_max, 
                                      min_val = 0, max_value = 1, title = 'data marginals', marginals = True)       
            self.plot_onepoint_values(onepoint = self.norm_marginals(self.gen_marginals), L_min=L_min, L_max=L_max, 
                                      min_val = 0, max_value = 1, title = 'generated marginals', marginals = True)
            self.plot_onepoint_values(onepoint = self.norm_marginals(self.model_marginals), L_min=L_min, L_max=L_max, 
                                      min_val = -8, max_value = 0, title = 'model marginals', marginals = True)