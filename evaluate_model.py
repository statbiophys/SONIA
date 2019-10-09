import numpy as np
import os
from sonia import Sonia
import multiprocessing as mp

class EvaluateModel(object):
	"""Class used to evaluate sequences with the sonia model


	Attributes
	----------
	Methods
	----------


	"""

	def __init__(self,sonia_model=None,olga_model=None,chain_type = 'human_T_beta',include_genes=True,processes=None):
		self.chain_type=chain_type
		self.define_olga_models(olga_model=olga_model)
		self.define_sonia_model(sonia_model=sonia_model)
		if 'v' in [s[0][0] for s in self.sonia_model.features]: self.include_genes=True
		else:self.include_genes=False
		if processes is None: self.processes = mp.cpu_count()


	def define_olga_models(self,olga_model=None):
		"""
		Defines Olga pgen and seqgen models and keeps them as attributes.

		"""
		import olga.load_model as load_model
		import olga.generation_probability as pgen
		import olga.sequence_generation as seq_gen


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
			main_folder=os.path.join(os.path.dirname(load_model.__file__), 'default_models', self.chain_type)

		params_file_name = os.path.join(main_folder,'model_params.txt')
		marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
		V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
		J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

		genomic_data = load_model.GenomicDataVDJ()
		genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
		self.genomic_data=genomic_data
		generative_model = load_model.GenerativeModelVDJ()
		generative_model.load_and_process_igor_model(marginals_file_name)        

		self.pgen_model = pgen.GenerationProbabilityVDJ(generative_model, genomic_data)
		self.pgen_model.V_mask_mapping=self.complement_V_mask(self.pgen_model)

		self.seq_gen_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)

	def define_sonia_model(self,sonia_model=None):
		"""
		Loads a Sonia model and keeps it as attribute.

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
		'''
		returns selection factors, pgen and pposts of sequences. 
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
		'''
		returns selection factors of sequences. 
		'''
		
		self.compute_energies(energies_data=False)
		
		#find seq features
		seq_features = [self.sonia_model.find_seq_features(seq) for seq in seqs]

		# compute energies
		energies =self.sonia_model.compute_energy(seq_features)
		
		return energies

	def rejection_vector(self,upper_bound=10,energies=None):
		'''
		Compute rejection 
		
		Parameters
		----------
		upper_bound : int or float
		accept all above the threshold (domain of validity of the model)
		
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
		custom_model_folder : str
			Path to a folder specifying a custom IGoR formatted model to be
			used as a generative model. Folder must contain 'model_params.txt'
			and 'model_marginals.txt'

		Returns
		--------------
		seqs : list
			MonteCarlo sequences drawn from a VDJ recomb model

		"""
		#Generate sequences
		#seqs_generated=generate_all_seqs(int(num_seqs),sg_model) # parallel version
		seqs_generated=[self.seq_gen_model.gen_rnd_prod_CDR3() for i in range(int(num_seqs))]
		seqs = [[seq[1], self.genomic_data.genV[seq[2]][0].split('*')[0], self.genomic_data.genJ[seq[3]][0].split('*')[0]] for seq in seqs_generated]#[sg_model.gen_rnd_prod_CDR3() for _ in range(int(num_gen_seqs))]]
		if not self.include_genes:
			seqs=list(np.array(seqs)[:,0])
			seqs= [[d,[],[]] for d in seqs]
		return seqs
	
	def generate_sequences_post(self,num_seqs,upper_bound=10):
		"""Generates MonteCarlo sequences from Sonia thourgh rejection sampling


		Parameters
		----------
		num_seqs : int or float
			Number of MonteCarlo sequences to generate and add to the specified
			sequence pool.
		upper_bound: int
			1/ratio of sequences that are rejected in the process 

		Returns
		--------------
		seqs : list
			MonteCarlo sequences drawn from a VDJ recomb model

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
		Compute energies for all sequences 
		'''
		
		self.sonia_model.gauge_energies() # set gauge for proper energies
		
		if energies_data:
			self.energies_data=self.sonia_model.compute_energy(self.sonia_model.data_seq_features)
			self.Q_data=np.exp(-self.energies_data)

		if energies_gen:
			self.energies_gen=self.sonia_model.compute_energy(self.sonia_model.gen_seq_features)
			self.Q_gen=np.exp(-self.energies_gen)
			self.Z=np.sum(self.Q_gen)/len(self.Q_gen)  

		 
	def compute_pgen(self,rejection_bound=10,custom_model_folder=None,default_olga=False):
		'''
		Compute pgen for all seqs in the dataset in parallel
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
		'''
		plt.figure(figsize=(12,8))
		binning_=np.linspace(-20,-5,n_bins)

		plt.hist(np.nan_to_num(np.log10(self.pgen_data)),binning_,histtype='step',normed=True,label='data')
		plt.hist(np.nan_to_num(np.log10(self.pgen_gen)),binning_,histtype='step',normed=True,label='pre-sel')
		plt.hist(np.nan_to_num(np.log10(self.pgen_sel)),binning_,histtype='step',normed=True,label='post-sel')

		plt.xlabel('$log_{10} P_{pre}$',fontsize=20)
		plt.ylabel('density',fontsize=20)
		plt.legend()
		plt.show()
		
	def plot_ppost(self,n_bins=100):
		'''
		Histogram plot of Ppost
		'''
		plt.figure(figsize=(12,8))
		binning_=np.linspace(-20,-5,n_bins)

		plt.hist(np.nan_to_num(np.log10(self.ppost_data)),binning_,histtype='step',normed=True,label='data')
		plt.hist(np.nan_to_num(np.log10(self.ppost_gen)),binning_,histtype='step',normed=True,label='pre-sel')
		plt.hist(np.nan_to_num(np.log10(self.ppost_sel)),binning_,histtype='step',normed=True,label='post-sel')

		plt.xlabel('$log_{10} P_{post}$',fontsize=20)
		plt.ylabel('density',fontsize=20)
		plt.legend()
		plt.show()


# some parallel utils for pgen computation

def compute_pgen_expand(x):
	return x[1].compute_aa_CDR3_pgen(x[0][0],x[0][1],x[0][2])

def compute_pgen_expand_novj(x):
	return x[1].compute_aa_CDR3_pgen(x[0][0])

def compute_all_pgens(seqs,model=None,processes=None,include_genes=True):
	'''
	Compute Pgen of sequences using OLGA
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