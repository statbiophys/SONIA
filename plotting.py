import numpy as np
import os
import matplotlib.pyplot as plt

class Plotter(object):
	"""Class used to plot stuff


	Attributes
	----------
	sonia_model: object
		Sonia model. No path.

	Methods
	----------
	
	plot_model_learning(save_name = None)
		Plots L1 convergence curve and marginal scatter.

	plot_pgen(pgen_data=[],pgen_gen=[],pgen_model=[],n_bins=100)
		Histogram plot of pgen. You need to evalute them first.

	plot_ppost(ppost_data=[],ppost_gen=[],pppst_model=[],n_bins=100)
		Histogram plot of ppost. You need to evalute them first.

	plot_model_parameters(low_freq_mask = 0.0)
		For LengthPos model only. Plot the model parameters using plot_onepoint_values

	plot_marginals_length_corrected(min_L = 8, max_L = 16, log_scale = True)
		For LengthPos model only. Plot length normalized marginals.

	"""

	def __init__(self,sonia_model=None):
		if type(sonia_model)==str or sonia_model is None: 
			print('ERROR: you need to pass a Sonia object')
			return
		self.sonia_model=sonia_model

	def plot_pgen(self,pgen_data=[],pgen_gen=[],pgen_model=[],n_bins=100,save_name=None):
		'''Histogram plot of Pgen

		Parameters
		----------
		n_bins: int
			number of bins of the histogram

		'''
		plt.figure(figsize=(12,8))
		binning_=np.linspace(-20,-5,n_bins)
		k,l=np.histogram(np.nan_to_num(np.log10(pgen_data)),binning_,density=True)
		plt.plot(l[:-1],k,label='data',linewidth=2)
		k,l=np.histogram(np.nan_to_num(np.log10(pgen_gen)),binning_,density=True)
		plt.plot(l[:-1],k,label='pre-sel',linewidth=2)
		k,l=np.histogram(np.nan_to_num(np.log10(pgen_sel)),binning_,density=True)
		plt.plot(l[:-1],k,label='post-sel',linewidth=2)

		plt.xlabel('$log_{10} P_{pre}$',fontsize=20)
		plt.ylabel('density',fontsize=20)
		plt.legend()
		fig.tight_layout()

		if save_name is not None:
			fig.savefig(save_name)
		plt.show()
		
	def plot_ppost(self,ppost_data=[],ppost_gen=[],pppst_model=[],n_bins=100,save_name=None):
		'''Histogram plot of Ppost

		Parameters
		----------
		n_bins: int
			number of bins of the histogram

		'''
		plt.figure(figsize=(12,8))
		binning_=np.linspace(-20,-5,n_bins)
		k,l=np.histogram(np.nan_to_num(np.log10(self.ppost_data)),binning_,density=True)
		plt.plot(l[:-1],k,label='data',linewidth=2)
		k,l=np.histogram(np.nan_to_num(np.log10(self.ppost_gen)),binning_,density=True)
		plt.plot(l[:-1],k,label='pre-sel',linewidth=2)
		k,l=np.histogram(np.nan_to_num(np.log10(self.ppost_sel)),binning_,density=True)
		plt.plot(l[:-1],k,label='post-sel',linewidth=2)

		plt.xlabel('$log_{10} P_{post}$',fontsize=20)
		plt.ylabel('density',fontsize=20)
		plt.legend()
		plt.show()

	def plot_model_learning(self, save_name = None):

		"""Plots L1 convergence curve and marginal scatter.

		Parameters
		----------
		save_name : str or None
			File name to save output figure. If None (default) does not save.

		"""

		min_for_plot = 1/(10.*np.power(10, np.ceil(np.log10(len(self.sonia_model.data_seqs)))))
		fig = plt.figure(figsize =(14, 4))
		fig.add_subplot(131)
		fig.subplots_adjust(left=0.1, bottom = 0.13, top = 0.91, right = 0.97, wspace = 0.3, hspace = 0.15)
		plt.loglog(range(1, len(self.sonia_model.L1_converge_history)+1), self.sonia_model.L1_converge_history, 'k', linewidth = 2)
		plt.xlabel('Iteration', fontsize = 13)
		plt.ylabel('L1 Distance', fontsize = 13)

		plt.legend(frameon = False, loc = 2)
		plt.title('L1 Distance convergence', fontsize = 15)
		
		fig.add_subplot(132)

		plt.loglog(self.sonia_model.data_marginals, self.sonia_model.gen_marginals, 'r.', alpha = 0.2, markersize=1)
		plt.loglog(self.sonia_model.data_marginals, self.sonia_model.model_marginals, 'b.', alpha = 0.2, markersize=1)
		plt.loglog([],[], 'r.', label = 'Raw marginals')
		plt.loglog([],[], 'b.', label = 'Model adjusted marginals')

		plt.loglog([min_for_plot, 2], [min_for_plot, 2], 'k--', linewidth = 0.5)
		plt.xlim([min_for_plot, 1])
		plt.ylim([min_for_plot, 1])

		plt.xlabel('Marginals over data', fontsize = 13)
		plt.ylabel('Marginals over generated sequences', fontsize = 13)
		plt.legend(loc = 2, fontsize = 10)
		plt.title('Marginal Scatter', fontsize = 15)
		
		fig.add_subplot(133)
		plt.title('Likelihood', fontsize = 15)
		plt.plot(self.sonia_model.learning_history.history['_likelihood'],label='train',c='k')
		plt.plot(self.sonia_model.learning_history.history['val__likelihood'],label='validation',c='r')
		plt.legend(fontsize = 10)
		plt.xlabel('Iteration', fontsize = 13)
		plt.ylabel('Likelihood', fontsize = 13)

		fig.tight_layout()

		if save_name is not None:
			fig.savefig(save_name)

		plt.show()

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
		
		for a,aa in enumerate(self.sonia_model.amino_acids):
		
			M = np.empty((max_L - min_L + 1, max_L))
			M[:] = np.nan
		
			for l in range(min_L, max_L + 1):
				for i in range(l):
					if onepoint_dict == None:
						M[l-min_L,i] = onepoint[self.sonia_model.feature_dict[('l' + str(l), 'a' + aa + str(i))]]
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
		p1 = np.exp(-self.sonia_model.model.get_weights()[0].flatten())
		if low_freq_mask:
			p1[(self.sonia_model.data_marginals < low_freq_mask) & (self.sonia_model.gen_marginals < low_freq_mask)] = -1
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
				for aa in self.sonia_model.amino_acids:
					if marg[self.sonia_model.feature_dict[('l' + str(l),)]]>0:
						marg[self.sonia_model.feature_dict[('l' + str(l), 'a' + aa + str(i))]] /= marg[self.sonia_model.feature_dict[('l' + str(l),)]]

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
			self.plot_onepoint_values(onepoint = np.log(self.norm_marginals(self.sonia_model.data_marginals) + pc), min_L=min_L, max_L=max_L ,
									  min_val = -8, max_value = 0, title = 'log(data marginals)', marginals = True)
			self.plot_onepoint_values(onepoint = np.log(self.norm_marginals(self.sonia_model.gen_marginals) + pc), min_L=min_L, max_L=max_L, 
									  min_val = -8, max_value = 0, title = 'log(generated marginals)', marginals = True)
			self.plot_onepoint_values(onepoint = np.log(self.norm_marginals(self.sonia_model.model_marginals) + pc), min_L=min_L, max_L=max_L, 
									  min_val = -8, max_value = 0, title = 'log(model marginals)', marginals = True)
		else:
			self.plot_onepoint_values(onepoint = self.norm_marginals(self.sonia_model.data_marginals), min_L=min_L, max_L=max_L, 
									  min_val = 0, max_value = 1, title = 'data marginals', marginals = True)       
			self.plot_onepoint_values(onepoint = self.norm_marginals(self.sonia_model.gen_marginals), min_L=min_L, max_L=max_L, 
									  min_val = 0, max_value = 1, title = 'generated marginals', marginals = True)
			self.plot_onepoint_values(onepoint = self.norm_marginals(self.sonia_model.model_marginals), min_L=min_L, max_L=max_L, 
									  min_val = -8, max_value = 0, title = 'model marginals', marginals = True)