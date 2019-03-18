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
    
    def __init__(self, features = [], constant_features = [], data_seqs = [], gen_seqs = [], chain_type = 'humanTRB', load_model = None, include_genes = True):
        
        QmodelCDR3.__init__(self, features, constant_features, data_seqs, gen_seqs, chain_type, load_model)
        self.min_L = min([len(x[0]) for x in (self.gen_seqs + self.data_seqs)])
        self.max_L = max([len(x[0]) for x in (self.gen_seqs + self.data_seqs)])
        self.add_features()
    
    def add_features(self, min_L = None, max_L = None, include_genes = True):
        """Generates a list of feature_strs for a model that parametrize every amino acid
        by the CDR3 length and position from the left (Cys)
        
        
        Parameters
        ----------
        min_L : int
            Minimum length CDR3 sequence
        max_L : int
            Maximum length CDR3 sequence
        include_genes : bool
            If true, features for gene selection are also generated
                
        """
        
        if min_L == None:
            min_L = self.min_L
        if max_L == None:
            max_L = self.max_L
        
        self.amino_acids =  'ARNDCQEGHILKMFPSTWYV'
        features = []
        L_features = [['l' + str(L)] for L in range(min_L, max_L + 1)]
        features += L_features
        for L in range(min_L, max_L + 1):
            for i in range(L):
                for aa in self.amino_acids:
                     features.append(['l' + str(L), 'a' + aa + str(i)])
                    
        if include_genes:
            main_folder = os.path.join(os.path.dirname(load_model.__file__), 'default_models', self.chain_type)
        
            params_file_name = os.path.join(main_folder,'model_params.txt')
            V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
            J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')
            
            genomic_data = load_model.GenomicDataVDJ()
            genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
            
            features += [[v, j] for v in set(['v' + genV[0].split('*')[0].split('V')[-1] for genV in genomic_data.genV]) for j in set(['j' + genJ[0].split('*')[0].split('J')[-1] for genJ in genomic_data.genJ])]
            
        self.update_model(add_features=features, add_constant_features=L_features)
        
        self.feature_dict = {tuple(x):i for i,x in enumerate(self.features)}
        
    def set_gauge(self, min_L = None, max_L = None):
        """ multiply all paramaters of the same position and length (all amino acids) by a common factor
        so sum_aa(gen_marginal(aa) * model_paramaters(aa)) = 1
        
        
        Parameters
        ----------
        min_L : int
            Minimum length CDR3 sequence, if not given taken from class attribute
        max_L : int
            Maximum length CDR3 sequence, if not given taken from class attribute
                
        """
        if min_L == None:
            min_L = self.min_L
        if max_L == None:
            max_L = self.max_L
        
        #min_L = min([int(x.split('_')[0][1:]) for x in self.features if x[0]=='L'])
        #max_L = max([int(x.split('_')[0][1:]) for x in self.features if x[0]=='L'])
        
        for l in range(min_L, max_L + 1):
            if self.gen_marginals[self.feature_dict[('l' + str(l),)]]>0:
                for i in range(l):
                    G = sum([self.gen_marginals[self.feature_dict[('l' + str(l), 'a' + aa + str(i))]]
                                    /self.gen_marginals[self.feature_dict[('l' + str(l),)]] * 
                                    np.exp(self.model_params[self.feature_dict[('l' + str(l), 'a' + aa + str(i))]]) 
                                    for aa in self.amino_acids])
                    for aa in self.amino_acids:
                        self.model_params[self.feature_dict[('l' + str(l), 'a' + aa + str(i))]] -= np.log(G)   
                        
    def plot_onepoint_values(self, onepoint = None ,onepoint_dict = None,  min_L = None, max_L = None, min_val = None, max_value = None, 
                             title = '', cmap = 'seismic', bad_color = 'black', aa_color = 'white', marginals = False):
        """ plot a function of aa, length and position from left, one heatplot per aa
        
        
        Parameters
        ----------
        onepoint : ndarray
            array containting one-point values to plot, in the same shape as self.features, 
            expected unless onepoint_dict is given
        onepoint_dict : dict
            dict of the one-point values to plot, keyed by the feature tuples such as (l12,aA8)
        min_L : int
            Minimum length CDR3 sequence
        max_L : int
            Maximum length CDR3 sequence
        min_val : float
            minimum value to plot
        max_val : float
            maximum value to plot
        title : string
            title of plot to display
        cmap : colormap 
            colormap to use for the heatplots
        bad_color : string
            color to use for nan values - used primarly for cells where position is larger than length
        aa_color : string
            color to use for amino acid names for each heatplot displayed on the bad_color background
        marginals : bool
            if true, indicates marginals are to be plotted and this sets cmap, bad_color and aa_color
        
        """
        
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
        
            M = np.empty((max_L - min_L + 1, max_L))
            M[:] = np.nan
        
            for l in range(min_L, max_L + 1):
                for i in range(l):
                    if onepoint_dict == None:
                        M[l-min_L,i] = onepoint[self.feature_dict[('l' + str(l), 'a' + aa + str(i))]]
                    else:
                        M[l-min_L,i] = onepoint_dict.get(('l' + str(l), 'a' + aa + str(i)), np.nan)
            
            im = grid[a].imshow(M, cmap = current_cmap, vmin = min_val, vmax = max_value)
            grid[a].text(0.75,0.7,amino_acids_dict[aa],transform=grid[a].transAxes, color = aa_color, fontsize = 'large', fontweight = 'bold')
    
        grid.cbar_axes[0].colorbar(im)


        
        grid.axes_llc.set_xticks(range(0, max_L, 2))
        grid.axes_llc.set_xticklabels(range(1, max_L + 1, 2))
        grid.axes_llc.set_yticks(range(0, max_L - min_L + 1 ))
        grid.axes_llc.set_yticklabels(range(min_L, max_L + 1))
        
        fig.suptitle(title, fontsize=20.00)

    def plot_model_parameters(self, low_freq_mask = 0.0):
        """ plot the model parameters using plot_onepoint_values
        
        Parameters
        ----------
        low_freq_mask : float
            threshold on the marginals, anything lower would be grayed out
        
        """
        p1 = np.exp(-self.model_params)
        if low_freq_mask:
            p1[(self.data_marginals < low_freq_mask) & (self.gen_marginals < low_freq_mask)] = -1
        self.plot_onepoint_values(onepoint = p1, min_L = 8, max_L = 16, min_val = 0, max_value = 2, title = 'model parameters q=exp(-E)')
        
    def norm_marginals(self, marg, min_L = None, max_L = None):
        """ renormalizing the marginals accourding to length, so the sum of the marginals over all amino acid 
            for one position/length combination will be 1 (and not the fraction of CDR3s of this length)
            
        Parameters
        ----------
        marg : ndarray
            the marginal to renormalize
        min_L : int
            Minimum length CDR3 sequence, if not given taken from class attribute
        max_L : int
            Maximum length CDR3 sequence, if not given taken from class attribute
                   
        
        """
        
        if min_L == None:
            min_L = self.min_L
        if max_L == None:
            max_L = self.max_L
            
        for l in range(min_L, max_L + 1):
            for i in range(l):
                for aa in self.amino_acids:
                    if marg[self.feature_dict[('l' + str(l),)]]>0:
                        marg[self.feature_dict[('l' + str(l), 'a' + aa + str(i))]] /= marg[self.feature_dict[('l' + str(l),)]]
        #length_correction = np.array([(marg[self.feature_dict[f.split('_')[0]]] if ('_' in f) else 1.0) for f in self.features])
        #length_correction[length_correction == 0] = 1 #avoid dividing by zero if there is no seq of this length
        return marg 
        

    def plot_marginals_length_corrected(self, min_L = 8, max_L = 16, log_scale = True):
        
        """ plot length normalized marginals using plot_onepoint_values
        
        Parameters
        ----------
        min_L : int
            Minimum length CDR3 sequence, if not given taken from class attribute
        max_L : int
            Maximum length CDR3 sequence, if not given taken from class attribute
        log_scale : bool
            if True (default) plots marginals on a log scale
        """
        
        if log_scale:
            pc = 1e-10 #pseudo count to add to marginals to avoid log of zero
            self.plot_onepoint_values(onepoint = np.log(self.norm_marginals(self.data_marginals) + pc), min_L=min_L, max_L=max_L ,
                                      min_val = -8, max_value = 0, title = 'log(data marginals)', marginals = True)
            self.plot_onepoint_values(onepoint = np.log(self.norm_marginals(self.gen_marginals) + pc), min_L=min_L, max_L=max_L, 
                                      min_val = -8, max_value = 0, title = 'log(generated marginals)', marginals = True)
            self.plot_onepoint_values(onepoint = np.log(self.norm_marginals(self.model_marginals) + pc), min_L=min_L, max_L=max_L, 
                                      min_val = -8, max_value = 0, title = 'log(model marginals)', marginals = True)
        else:
            self.plot_onepoint_values(onepoint = self.norm_marginals(self.data_marginals), min_L=min_L, max_L=max_L, 
                                      min_val = 0, max_value = 1, title = 'data marginals', marginals = True)       
            self.plot_onepoint_values(onepoint = self.norm_marginals(self.gen_marginals), min_L=min_L, max_L=max_L, 
                                      min_val = 0, max_value = 1, title = 'generated marginals', marginals = True)
            self.plot_onepoint_values(onepoint = self.norm_marginals(self.model_marginals), min_L=min_L, max_L=max_L, 
                                      min_val = -8, max_value = 0, title = 'model marginals', marginals = True)