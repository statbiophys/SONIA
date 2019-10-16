import numpy as np
import os
from sonia import Sonia
import multiprocessing as mp
import matplotlib.pyplot as plt
import olga.load_model as olga_load_model
import olga.generation_probability as pgen
import olga.sequence_generation as seq_gen

class EvaluateModel(object):
	"""Class used to evaluate sequences with the sonia model


	Attributes
	----------
	sonia_model: object , (optionally string but not recommended)
		Sonia model.
		Alternatively, path to a folder specifying a Sonia model.
		The second option is not suggested, since it is much slower.

	olga_model: string
		Path to a folder specifying a custom IGoR formatted model to be
		used as a generative model. Folder must contain 'model_params.txt',
		model_marginals.txt','V_gene_CDR3_anchors.csv' and 'J_gene_CDR3_anchors.csv'.

	chain_type: string
		Type of receptor. This specification is used to determine gene names
		and allow integrated OLGA sequence generation. Options: 'humanTRA',
		'humanTRB' (default), 'humanIGH', and 'mouseTRB'.

	include_genes: bool
		Presence of gene information in data/generated sequences

	processes: int
		number of processes to use to infer pgen. Deafult: all.

	Methods
	----------
	define_olga_models(olga_model=None)
		Defines Olga pgen and seqgen models and keeps them as attributes.

	define_sonia_model(sonia_model=None)
		Imports a Sonia model and keeps it as attribute.
	
	complement_V_mask(model)
		add V genes to OLGA mask.

	evaluate_seqs(seqs=[])
		Returns energies, pgen and ppost of a list of sequences. 

	evaluate_energies_seqs(seqs=[])
		Returns energies of a list of sequences.
	
	rejection_vector(upper_bound=10,energies=None)
		Returns acceptance from rejection sampling of a list of seqs.
		By default uses the generated sequences within the sonia model.

	generate_sequences_pre(num_seqs = 1)
		Generate sequences using olga
	
	generate_sequences_post(num_seqs,upper_bound=10)
		Generate sequences using olga and perform rejection selection.

	compute_energies(energies_gen=True,energies_data=True)
		Compute energies of data and generated seqs in the sonia model.
	
	compute_pgen(rejection_bound=10)
		Compute pgen of data and generated seqs in the sonia model in parallel.
	
	compute_ppost()
		Compute ppost of data and generated seqs in the sonia model
		by weighting the pgen with selection factor Qs=exp(-energies)

	plot_pgen(n_bins=100)
		Histogram plot of pgen

	plot_ppost(n_bins=100)
		Histogram plot of ppost

	reject_bad_features(threshold=5)
		Keeps only the features associated with marginals that have a high enough 
		count in the gen pool. Restricted only to VJ genes.

	"""

	def __init__(self,sonia_model=None,olga_model=None,chain_type = 'human_T_beta',include_genes=True,processes=None):
		default_chain_types = {'humanTRA': 'human_T_alpha', 'human_T_alpha': 'human_T_alpha', 'humanTRB': 'human_T_beta', 'human_T_beta': 'human_T_beta', 'humanIGH': 'human_B_heavy', 'human_B_heavy': 'human_B_heavy', 'mouseTRB': 'mouse_T_beta', 'mouse_T_beta': 'mouse_T_beta'}
		if chain_type not in default_chain_types.keys():
			print 'Unrecognized chain_type (not a default OLGA model). Please specify one of the following options: humanTRA, humanTRB, humanIGH, or mouseTRB.'
			return None
		self.chain_type = default_chain_types[chain_type]

		self.define_olga_models(olga_model=olga_model)
		self.define_sonia_model(sonia_model=sonia_model)
		self.include_genes=include_genes
		if processes is None: self.processes = mp.cpu_count()


	def define_olga_models(self,olga_model=None):
		"""Defines Olga pgen and seqgen models and keeps them as attributes.

		Parameters
		----------
		olga_model: string
			Path to a folder specifying a custom IGoR formatted model to be
			used as a generative model. Folder must contain 'model_params.txt',
			model_marginals.txt','V_gene_CDR3_anchors.csv' and 'J_gene_CDR3_anchors.csv'.


		Attributes set
		--------------
		genomic_data: object
			genomic data associate with the olga model.

		pgen_model: object
			olga model for evaluation of pgen.

		seq_gen_model: object
			olga model for generation of seqs.

		"""


		#Load generative model
		if olga_model is not None:
			try:
				# relative path
				pathdir= os.getcwd()
				main_folder = os.path.join(pathdir,olga_model)
				os.path.isfile(os.path.join(main_folder,'model_params.txt'))
			except:
				# absolute path
				main_folder=olga_model
		else:
			main_folder=os.path.join(os.path.dirname(olga_load_model.__file__), 'default_models', self.chain_type)

		params_file_name = os.path.join(main_folder,'model_params.txt')
		marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
		V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
		J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

		genomic_data = olga_load_model.GenomicDataVDJ()
		genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
		self.genomic_data=genomic_data
		generative_model = olga_load_model.GenerativeModelVDJ()
		generative_model.load_and_process_igor_model(marginals_file_name)        

		self.pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)
		self.pgen_model.V_mask_mapping=self.complement_V_mask(self.pgen_model)

		self.seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)

	def define_sonia_model(self,sonia_model=None):
		"""Loads a Sonia model and keeps it as attribute.

		Parameters
		----------
		sonia_model: object , (optionally string but not recommended)
			Sonia model.
			Alternatively, path to a folder specifying a Sonia model.
			The second option is not suggested, since it is much slower.

		Attributes set
		--------------
		sonia_model: object
			Sonia model.


		"""
		try:
			# load from file, generic load, means the find_seq_feats function is slow.
			self.sonia_model=Sonia(load_model=sonia_model)
		except:
			# sonia model passed as an argument
			self.sonia_model=sonia_model

	def complement_V_mask(self,model):
		"""Add V genese with -1 at end. 
		i.e before TRBV9 only, then TRBV9 and TRBV9-1  

		Parameters
		----------
		model: object
			Olga model for evaluation of pgen

		Returns
		-------
		V_mask_mapping: dict
			Dictionary that maps V genes to olga model parameters.

		"""
		x,y=[],[]
		for f in model.V_mask_mapping:
			x.append(f)
			y.append(model.V_mask_mapping[f])
			if (not '-' in f) and (not '*' in f):
				x.append(f+'-1')
				y.append(model.V_mask_mapping[f])
		return dict(zip(x,y))

	def evaluate_seqs(self,seqs=[]):
		'''Returns energies, pgen and pposts of sequences. 

		Parameters
		----------
		seqs: list
			list of sequences to evaluate

		Returns
		-------
		energies: array
			energies,i.e. -log(Q), of the sequences

		pgens: array
			pgen of the sequences

		pposts: array
			ppost of the sequences
		'''
		
		self.compute_energies(energies_data=False)
		
		#find seq features
		seq_features = [self.sonia_model.find_seq_features(seq) for seq in seqs]

		# compute energies
		energies =self.sonia_model.compute_energy(seq_features)
		
		# compute pgen
		pgens=compute_all_pgens(seqs,self.pgen_model,self.processes,self.include_genes)
		
		# compute ppost
		pposts=pgens*np.exp(-energies)/self.Z
		
		return energies, pgens, pposts
	
	def evaluate_energies_seqs(self,seqs=[]):
		'''Returns energies of sequences. 

		Parameters
		----------
		seqs: list
			list of sequences to evaluate

		Returns
		-------
		energies: array
			energies,i.e. -log(Q), of the sequences

		'''
		
		self.compute_energies(energies_data=False)
		
		#find seq features
		seq_features = [self.sonia_model.find_seq_features(seq) for seq in seqs]

		# compute energies
		energies =self.sonia_model.compute_energy(seq_features)
		
		return energies

	def rejection_vector(self,upper_bound=10,energies=None):
		''' Returns acceptance from rejection sampling of a list of seqs.
		By default uses the generated sequences within the sonia model.
		
		Parameters
		----------
		upper_bound : int or float
			accept all above the threshold. Relates to the percentage of 
			sequences that pass selection

		Returns
		-------
		rejection selection: array of bool
			acceptance of each sequence.
		
		'''

		self.compute_energies(energies_data=False) # you need Z
		if energies is None: energies=self.energies_gen

		# sample from uniform distribution
		random_samples=np.random.uniform(size=len(energies))
		Q=np.exp(-energies)

		#rejection vector
		self.rejection_selection=random_samples < np.clip(Q/self.Z,0,upper_bound)/float(upper_bound)

		#print 'acceptance frequency:',np.sum(self.rejection_selection)/float(len(self.rejection_selection))
		return self.rejection_selection

	def generate_sequences_pre(self, num_seqs = 1):
		"""Generates MonteCarlo sequences for gen_seqs using OLGA.

		Only generates seqs from a V(D)J model. Requires the OLGA package
		(pip install olga).

		Parameters
		----------
		num_seqs : int or float
			Number of MonteCarlo sequences to generate and add to the specified
			sequence pool.
		
		Returns
		--------------
		seqs : list
			MonteCarlo sequences drawn from a VDJ recomb model

		"""
		#Generate sequences
		seqs_generated=[self.seq_gen_model.gen_rnd_prod_CDR3() for i in range(int(num_seqs))]
		seqs = [[seq[1], self.genomic_data.genV[seq[2]][0].split('*')[0], self.genomic_data.genJ[seq[3]][0].split('*')[0]] for seq in seqs_generated]#[sg_model.gen_rnd_prod_CDR3() for _ in range(int(num_gen_seqs))]]
		if not self.include_genes:
			seqs=list(np.array(seqs)[:,0])
			seqs= [[d,[],[]] for d in seqs]
		return seqs
	
	def generate_sequences_post(self,num_seqs,upper_bound=10):
		"""Generates MonteCarlo sequences from Sonia through rejection sampling.

		Parameters
		----------
		num_seqs : int or float
			Number of MonteCarlo sequences to generate and add to the specified
			sequence pool.
		upper_bound: int
			accept all above the threshold. Relates to the percentage of 
			sequences that pass selection.

		Returns
		--------------
		seqs : list
			MonteCarlo sequences drawn from a VDJ recomb model that pass selection.

		"""
		seqs_post=[['a','b','c']] # initialize

		while len(seqs_post)<num_seqs:
			# generate sequences from pre
			seqs=self.generate_sequences_pre(num_seqs = int(1.5*upper_bound*num_seqs))

			# compute features and energies 
			seq_features = [self.sonia_model.find_seq_features(seq) for seq in seqs]
			energies = self.sonia_model.compute_energy(seq_features)
			rejection_selection=self.rejection_vector(upper_bound=upper_bound,energies=energies)
			seqs_post=np.concatenate([seqs_post,np.array(seqs)[rejection_selection]])
		return seqs_post[1:num_seqs+1]

	def compute_energies(self,energies_gen=True,energies_data=True):
		'''
		Compute energies of data and generated seqs in the sonia model.


		Parameters
		----------
		energies_gen: bool
			Energies are computed on generated data

		energies_data: bool
			Energies are computed on data

		Attributes set
		--------------
		
		energies_data: array
		Q_data: array
		energies_gen: array
		Q_gen: array
		Z:

		'''
				


		if energies_gen:
			self.energies_gen=self.sonia_model.compute_energy(self.sonia_model.gen_seq_features)
			self.Q_gen=np.exp(-self.energies_gen)
			self.Z=np.sum(self.Q_gen)/len(self.Q_gen)  

		if energies_data:
			self.energies_data=self.sonia_model.compute_energy(self.sonia_model.data_seq_features)
			self.Q_data=np.exp(-self.energies_data)/self.Z

		 
	def compute_pgen(self,rejection_bound=10):
		'''
		Compute pgen for all seqs in the dataset in parallel

		Parameters
		----------

		Attributes set
		--------------

		Returns
		-------

		'''

		try: self.pgen_data
		except: self.pgen_data=compute_all_pgens(self.sonia_model.data_seqs,self.pgen_model,self.processes,self.include_genes) # multiprocessor version
		try: self.pgen_gen
		except: self.pgen_gen=compute_all_pgens(self.sonia_model.gen_seqs,self.pgen_model,self.processes,self.include_genes) # multiprocessor version
			
		self.compute_energies()
		self.rejection_vector(rejection_bound)
		self.pgen_sel=np.array(self.pgen_gen)[self.rejection_selection] #add energies
	
	def compute_ppost(self):
		'''
		Compute ppost by weighting with selection factor Q=exp(-E)

		Parameters
		----------

		Attributes set
		--------------

		Returns
		-------
		'''
		self.compute_energies()
		self.Q_gen=np.exp(-self.energies_gen)
		self.Z=np.sum(self.Q_gen)/len(self.Q_gen)

		self.ppost_data=self.pgen_data*np.exp(-self.energies_data)/self.Z
		self.ppost_gen=self.pgen_gen*np.exp(-self.energies_gen)/self.Z
		self.ppost_sel=np.array(self.ppost_gen)[self.rejection_selection] #add energies
	
	
	def plot_pgen(self,n_bins=100):
		'''
		Histogram plot of Pgen

		Parameters
		----------

		Attributes set
		--------------

		Returns
		-------
		'''
		plt.figure(figsize=(12,8))
		binning_=np.linspace(-20,-5,n_bins)
		k,l=np.histogram(np.nan_to_num(np.log10(self.pgen_data)),binning_,density=True)
		plt.plot(l[:-1],k,label='data',linewidth=2)
		k,l=np.histogram(np.nan_to_num(np.log10(self.pgen_gen)),binning_,density=True)
		plt.plot(l[:-1],k,label='pre-sel',linewidth=2)
		k,l=np.histogram(np.nan_to_num(np.log10(self.pgen_sel)),binning_,density=True)
		plt.plot(l[:-1],k,label='post-sel',linewidth=2)

		plt.xlabel('$log_{10} P_{pre}$',fontsize=20)
		plt.ylabel('density',fontsize=20)
		plt.legend()
		plt.show()
		
	def plot_ppost(self,n_bins=100):
		'''
		Histogram plot of Ppost

		Parameters
		----------

		Attributes set
		--------------

		Returns
		-------
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

	def reject_bad_features(self,threshold=5):
		""" Keeps only the features associated with marginals that have a high enough count in the gen pool.
		Now restricted only to VJ genes.

		Parameters
		----------
		threshold : int
			minimum number of counts in datasets

		Attributes set
		----------
		features: list

		"""
		if not self.include_genes: return True # skip rejection if don't have v,j genes
		self.sonia_model.gen_marginals = self.sonia_model.compute_marginals(seq_model_features = self.sonia_model.gen_seq_features, use_flat_distribution = True)
		n_gen=(self.sonia_model.gen_marginals*len(self.sonia_model.gen_seq_features)).astype(np.int) # get counts
		selection=n_gen>threshold
		selection[:np.sum([(q[0][0]=='a' or q[0][0]=='l')for q in self.sonia_model.features])]=True # throw away only vj bad components
		self.sonia_model.features=self.sonia_model.features[selection]
		self.sonia_model.feature_dict = {tuple(f): i for i, f in enumerate(self.sonia_model.features)}
		self.sonia_model.update_model_structure(initialize=True)
		self.sonia_model.update_model(auto_update_seq_features=True)

		return True 

# some parallel utils for pgen computation

def compute_pgen_expand(x):
	return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj(x):
	return x[1].compute_aa_CDR3_pgen(x[0][0])

def compute_all_pgens(seqs,model=None,processes=None,include_genes=True):
	'''Compute Pgen of sequences using OLGA

	Parameters
	----------


	Returns
	-------
	'''
	#Load OLGA for seq pgen estimation
	if model is None:
		import olga.load_model as load_model
		import olga.generation_probability as pgen

		main_folder = os.path.join(os.path.dirname(load_model.__file__), 'default_models', chain_type)
		params_file_name = os.path.join(main_folder,'model_params.txt')
		marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
		V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
		J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

		genomic_data = load_model.GenomicDataVDJ()
		genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
		generative_model = load_model.GenerativeModelVDJ()
		generative_model.load_and_process_igor_model(marginals_file_name)        
		model_pgen = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)
	
	# every process needs to access this vector, for sure there is a smarter way to implement this.
	final_models = [model for i in range(len(seqs))]

	pool = mp.Pool(processes=processes)
	if include_genes: 
		f=pool.map(compute_pgen_expand, zip(seqs,final_models))
		pool.close()
		return f
	else: 
		f=pool.map(compute_pgen_expand_novj, zip(seqs,final_models))
		pool.close()
		return f