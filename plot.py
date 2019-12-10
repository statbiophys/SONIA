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

	plot_pgen(pgen_data=[],pgen_gen=[],pgen_model=[],n_bins=100)
		Histogram plot of pgen

	plot_ppost(ppost_data=[],ppost_gen=[],pppst_model=[],n_bins=100)
		Histogram plot of ppost

	plot_model_learning(save_name = None)
		Plots L1 convergence curve and marginal scatter.

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
		
	def plot_ppost(self,ppost_data=[],ppost_gen=[],pppst_model=[],n_bins=100,save_name=None)
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
		plt.plot(self.sonia_model.learning_history.history['likelihood'],label='train',c='k')
		plt.plot(self.sonia_model.learning_history.history['val_likelihood'],label='validation',c='r')
		plt.legend(fontsize = 10)
		plt.xlabel('Iteration', fontsize = 13)
		plt.ylabel('Likelihood', fontsize = 13)

		fig.tight_layout()

		if save_name is not None:
			fig.savefig(save_name)

		plt.show()