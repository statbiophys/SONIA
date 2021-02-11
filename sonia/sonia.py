#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:06:58 2019

@author: zacharysethna
"""
from __future__ import print_function, division,absolute_import
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input,Dense,Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.backend import sum as ksum
from tensorflow.keras.backend import log as klog
from tensorflow.keras.backend import exp as kexp
from tensorflow.keras.backend import clip as kclip
import olga.load_model as olga_load_model
import olga.sequence_generation as seq_gen
from copy import copy
from tqdm import tqdm
#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass


class Sonia(object):
    """Class used to infer a Q selection model.


    Attributes
    ----------
    features : ndarray
        Array of feature lists. Each list contains individual subfeatures which
            all must be satisfied.
    features_dict : dict
        Dictionary keyed by tuples of the feature lists. Values are the index
        of the feature, i.e. self.features[self.features_dict[tuple(f)]] = f.
    constant_features : list
        List of feature strings to not update parameters during learning. These
            features are still used to compute energies (not currently used)
    data_seqs : list
        Data sequences used to infer selection model. Note, each 'sequence'
        is a list where the first element is the CDR3 sequence which is
        followed by any V or J genes.
    gen_seqs : list
        Sequences from generative distribution used to infer selection model.
        Note, each 'sequence' is a list where the first element is the CDR3
        sequence which is followed by any V or J genes.
    data_seq_features : list
        Lists of features that data_seqs project onto
    gen_seq_features : list
        Lists of features that gen_seqs project onto
    data_marginals : ndarray
        Array of the marginals of each feature over data_seqs
    gen_marginals : ndarray
        Array of the marginals of each feature over gen_seqs
    model_marginals : ndarray
        Array of the marginals of each feature over the model weighted gen_seqs
    L1_converge_history : list
        L1 distance between data_marginals and model_marginals at each
        iteration.
    chain_type : str
        Type of receptor. This specification is used to determine gene names
        and allow integrated OLGA sequence generation. Options: 'humanTRA',
        'humanTRB' (default), 'humanIGH', 'humanIGL', 'humanIGK' and 'mouseTRB'.
    l2_reg : float or None
        L2 regularization. If None (default) then no regularization.

    Methods
    ----------
    seq_feature_proj(feature, seq)
        Determines if a feature matches/is found in a sequence.

    find_seq_features(seq, features = None)
        Determines all model features of a sequence.

    compute_seq_energy(seq_features = None, seq = None)
        Computes the energy, as determined by the model, of a sequence.

    compute_energy(seqs_features)
        Computes the energies of a list of seq_features according to the model.

    compute_marginals(self, features = None, seq_model_features = None, seqs = None, use_flat_distribution = False)
        Computes the marginals of features over a set of sequences.

    infer_selection(self, epochs = 20, batch_size=5000, initialize = True, seed = None)
        Infers model parameters (energies for each feature).

    update_model_structure(self,output_layer=[],input_layer=[],initialize=False)
        Sets keras model structure and compiles.

    update_model(self, add_data_seqs = [], add_gen_seqs = [], add_features = [], remove_features = [], add_constant_features = [], auto_update_marginals = False, auto_update_seq_features = False)
        Updates model by adding/removing model features or data/generated seqs.
        Marginals and seq_features can also be updated.

    add_generated_seqs(self, num_gen_seqs = 0, reset_gen_seqs = True)
        Generates synthetic sequences using OLGA and adds them to gen_seqs.

    plot_model_learning(self, save_name = None)
        Plots current marginal scatter plot as well as L1 convergence history.

    save_model(self, save_dir, attributes_to_save = None)
        Saves the model.

    load_model(self, load_dir, load_seqs = True)
        Loads a model.

    """

    def __init__(self, features = [], data_seqs = [], gen_seqs = [], chain_type = 'humanTRB',
                 load_dir = None, feature_file = None, model_file = None, data_seq_file = None, gen_seq_file = None, log_file = None, load_seqs = True,
                 l2_reg = 0., l1_reg=0.,min_energy_clip = -5, max_energy_clip = 10, seed = None,vj=False):
        self.features = np.array(features)
        self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}
        self.data_seqs = []
        self.gen_seqs = []
        self.data_seq_features = []
        self.gen_seq_features = []
        self.data_marginals = np.zeros(len(features))
        self.gen_marginals = np.zeros(len(features))
        self.model_marginals = np.zeros(len(features))
        self.L1_converge_history = []
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.min_energy_clip = min_energy_clip
        self.max_energy_clip = max_energy_clip
        self.likelihood_train=[]
        self.likelihood_test=[]

        self.gamma=1.
        self.Z=1.
        default_chain_types = { 'humanTRA': 'human_T_alpha', 'human_T_alpha': 'human_T_alpha', 
                                'humanTRB': 'human_T_beta', 'human_T_beta': 'human_T_beta', 
                                'humanIGH': 'human_B_heavy', 'human_B_heavy': 'human_B_heavy', 
                                'humanIGK': 'human_B_kappa', 'human_B_kappa': 'human_B_kappa', 
                                'humanIGL': 'human_B_lambda', 'human_B_lambda': 'human_B_lambda', 
                                'mouseTRB': 'mouse_T_beta', 'mouse_T_beta': 'mouse_T_beta'}
        if chain_type not in default_chain_types.keys():
            print('Unrecognized chain_type (not a default OLGA model). Please specify one of the following options: humanTRA, humanTRB, humanIGH, humanIGK, humanIGL or mouseTRB.')
            return None
        self.chain_type = default_chain_types[chain_type]
        self.vj=vj
        if self.chain_type in ['human_T_alpha','human_B_kappa','human_B_lambda','mouse_T_alpha']: self.vj=True

        norms={'human_T_beta':0.2442847269027897,'human_T_alpha':0.2847166577727317,'human_B_heavy': 0.1499265655936305, 
                'human_B_lambda':0.29489499727399304, 'human_B_kappa':0.29247125650320943, 'mouse_T_beta':0.2727148540013573}
        self.norm_productive=norms[self.chain_type]

        if any([x is not None for x in [load_dir, feature_file, model_file]]):
            self.load_model(load_dir = load_dir, feature_file = feature_file, model_file = model_file, data_seq_file = data_seq_file, gen_seq_file = gen_seq_file, log_file = log_file, load_seqs = load_seqs)
            if len(self.data_seqs) == 0: self.update_model(add_data_seqs = data_seqs)
            if len(self.gen_seqs) == 0: self.update_model(add_data_seqs = gen_seqs)
        else:
            self.update_model(add_data_seqs = data_seqs, add_gen_seqs = gen_seqs)
            self.update_model_structure(initialize=True)
            self.L1_converge_history = []

        if seed is not None:
            np.random.seed(seed = seed)

        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    def seq_feature_proj(self, feature, seq):
        """Checks if a sequence matches all subfeatures of the feature list

        Parameters
        ----------
        feature : list
            List of individual subfeatures the sequence must match
        seq : list
            CDR3 sequence and any associated genes

        Returns
        -------
        bool
            True if seq matches feature else False.
        """

        try:
            for f in feature:
                if f[0] == 'a': #Amino acid subfeature
                    if len(f) == 2:
                        if f[1] not in seq[0]:
                            return False
                    elif seq[0][int(f[2:])] != f[1]:
                        return False
                elif f[0] == 'v' or f[0] == 'j': #Gene subfeature
                        if not any([[int(x) for x in f[1:].split('-')] == [int(y) for y in gene.lower().split(f[0])[-1].split('-')] for gene in seq[1:] if f[0] in gene.lower()]):
                            if not any([[int(x) for x in f[1:].split('-')] == [int(gene.lower().split(f[0])[-1].split('-')[0])] for gene in seq[1:] if f[0] in gene.lower()]): #Checks for gene-family match if specified as such
                                return False
                elif f[0] == 'l': #CDR3 length subfeature
                    if len(seq[0]) != int(f[1:]):
                        return False
        except: #All ValueErrors and IndexErrors return False
            return False

        return True

    def find_seq_features(self, seq, features = None):
        """Finds which features match seq

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
            features = self.features
        seq_features = []
        for feature_index, feature_lst in enumerate(features):
            if self.seq_feature_proj(feature_lst, seq):
                seq_features += [feature_index]

        return seq_features

    def compute_seq_energy(self, seq = None, seq_features = None):
        """Computes the energy of a sequence according to the model.

        Parameters
        ----------
        seq : list
            CDR3 sequence and any associated genes
        seq_features : list
            Features indices seq projects onto.

        Returns
        -------
        E : float
            Energy of seq according to the model.
        """
        if seq_features is not None:
            return self.compute_energy([seq_features])[0]
        elif seq is not None:
            return self.compute_energy([self.find_seq_features(seq)])[0]
        return 0

    def compute_energy(self,seqs_features):
        """Computes the energy of a list of sequences according to the model.

        Parameters
        ----------
        seqs_features : list
            list of encoded sequences into sonia features.

        Returns
        -------
        E : float
            Energies of seqs according to the model.

        """
        seqs_features_enc=self._encode_data(seqs_features)
        return self.model.predict(seqs_features_enc)[:, 0]

    def _encode_data(self,seq_features):
        """Turns seq_features into expanded numpy array"""

        if len(seq_features[0])==0: seq_features=[seq_features]
        length_input=len(self.features)
        data=np.array(seq_features, dtype=object)
        data_enc = np.zeros((len(data), length_input), dtype=np.int8)
        for i in range(len(data_enc)): data_enc[i][data[i]] = 1
        return data_enc


    def compute_marginals(self, features = None, seq_model_features = None, seqs = None, use_flat_distribution = False, output_dict = False):
        """Computes the marginals of each feature over sequences.
        Computes marginals either with a flat distribution over the sequences
        or weighted by the model energies. Note, finding the features of each
        sequence takes time and should be avoided if it has already been done.
        If computing marginals of model features use the default setting to
        prevent searching for the model features a second time. Similarly, if
        seq_model_features has already been determined use this to avoid
        recalculating it.

        Parameters
        ----------
        features : list or None
            List of features. This does not need to match the model
            features. If None (default) the model features will be used.
        seq_features_all : list
            Indices of model features seqs project onto.
        seqs : list
            List of sequences to compute the feature marginals over. Note, each
            'sequence' is a list where the first element is the CDR3 sequence
            which is followed by any V or J genes.
        use_flat_distribution : bool
            Marginals will be computed using a flat distribution (each seq is
            weighted as 1) if True. If False, the marginals are computed using
            model weights (each sequence is weighted as exp(-E) = Q). Default
            is False.

        Returns
        -------
        marginals : ndarray or dict
            Marginals of model features over seqs.

        """
        if seq_model_features is None:  # if model features are not given
            if seqs is not None:  # if sequences are given, get model features from them
                seq_model_features = [self.find_seq_features(seq) for seq in seqs]
            else:   # if no sequences are given, sequences features is empty
                seq_model_features = []

        if len(seq_model_features) == 0:  # if no model features are given, return empty array
            return np.array([])

        if features is not None and seqs is not None:  # if a different set of features for the marginals is given
            seq_compute_features = [self.find_seq_features(seq, features = features) for seq in seqs]
        else:  # if no features or no sequences are provided, compute marginals using model features
            seq_compute_features = seq_model_features
            features = self.features

        marginals = np.zeros(len(features))
        if len(seq_model_features) != 0:
            Z = 0.
            if not use_flat_distribution:
                energies = self.compute_energy(seq_model_features)
                Qs= np.exp(-energies)
                for seq_features,Q in zip(seq_compute_features,Qs):
                    marginals[seq_features] += Q
                    Z += Q
            else:
                for seq_features in seq_compute_features:
                    marginals[seq_features] += 1.
                    Z += 1.

            marginals = marginals / Z
        return marginals

    def infer_selection(self, epochs = 10, batch_size=5000, initialize = True, seed = None,validation_split=0.2, monitor=False,verbose=0):
        """Infer model parameters, i.e. energies for each model feature.

        Parameters
        ----------
        epochs : int
            Maximum number of learning epochs
        intialize : bool
            Resets data shuffle
        batch_size : int
            Size of the batches in the inference
        seed : int
            Sets random seed

        Attributes set
        --------------
        model : keras model
            Parameters of the model
        model_marginals : array
            Marginals over the generated sequences, reweighted by the model.
        L1_converge_history : list
            L1 distance between data_marginals and model_marginals at each
            iteration.

        """

        if seed is not None:
            np.random.seed(seed = seed)
        if initialize:
            # prepare data
            self.X = np.array(self.data_seq_features+self.gen_seq_features, dtype=object)
            self.Y = np.concatenate([np.zeros(len(self.data_seq_features)), np.ones(len(self.gen_seq_features))])

            shuffle = np.random.permutation(len(self.X)) # shuffle
            self.X=self.X[shuffle]
            self.Y=self.Y[shuffle]

        
        if monitor:
            from sonia.utils import computeL1
            computeL1_dist = computeL1(self)
            callbacks = [computeL1_dist]
        else: callbacks=[]
        
        self.learning_history = self.model.fit(self._encode_data(self.X), self.Y, epochs=epochs, batch_size=batch_size,
                                          validation_split=validation_split, verbose=verbose, callbacks=callbacks)
        self.likelihood_train=self.learning_history.history['_likelihood']
        self.likelihood_test=self.learning_history.history['val__likelihood']

        if monitor:
            self.L1_converge_history = computeL1_dist.L1_history
            self.model.set_weights(computeL1_dist.weights_cpt)
        
        # set Z    
        self.energies_gen=self.compute_energy(self.gen_seq_features)
        self.Z=np.sum(np.exp(-self.energies_gen))/len(self.energies_gen)
        self.set_gauge()
        
        self.update_model(auto_update_marginals=True)

    def set_gauge(self):
        '''
        placeholder for gauges.
        '''
        pass

    def update_model_structure(self,output_layer=[],input_layer=[],initialize=False):
        """ Defines the model structure and compiles it.

        Parameters
        ----------
        structure : Sequential Model Keras
            structure of the model

        initialize: bool
            if True, it initializes to linear model, otherwise it updates to new structure

        """
        length_input=np.max([len(self.features),1])
        min_clip=copy(self.min_energy_clip)
        max_clip=copy(self.max_energy_clip)
        l2_reg=copy(self.l2_reg)
        l1_reg=copy(self.l1_reg)

        if initialize:
            input_layer = Input(shape=(length_input,))
            output_layer = Dense(1,use_bias=False,activation='linear',kernel_regularizer=l1_l2(l2=l2_reg,l1=l1_reg))(input_layer) #normal glm model

        # Define model
        clipped_out=Lambda(lambda x: kclip(x,min_clip,max_clip))(output_layer)
        self.model = Model(inputs=input_layer, outputs=clipped_out)

        self.optimizer = RMSprop()
        self.model.compile(optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood])
        return True

    def _loss(self, y_true, y_pred):
        """Loss function for keras training. 
            We assume a model of the form P(x)=exp(-E(x))P_0(x)/Z.
            We minimize the neg-loglikelihood: <-logP> = log(Z) - <-E>.
            Normalization of P gives Z=<exp(-E)>_{P_0}.
            We fix the gauge by adding the constraint (Z-1)**2 to the likelihood.
        """
        data= ksum((-y_pred)*(1.-y_true))/ksum(1.-y_true)
        gen= klog(ksum(kexp(-y_pred)*y_true))-klog(ksum(y_true))
        reg= kexp(gen)-1.
        return gen-data+self.gamma*reg*reg

    def _likelihood(self, y_true, y_pred):
        """Loss function for keras training. 
            We assume a model of the form P(x)=exp(-E(x))P_0(x)/Z.
            We minimize the neg-loglikelihood: <-logP> = log(Z) - <-E>.
            Normalization of P gives Z=<exp(-E)>_{P_0}.z
        """
        data= ksum((-y_pred)*(1.-y_true))/ksum(1.-y_true)
        gen= klog(ksum(kexp(-y_pred)*y_true))-klog(ksum(y_true))
        return gen-data

    def update_model(self, add_data_seqs = [], add_gen_seqs = [], add_features = [], remove_features = [], add_constant_features = [], auto_update_marginals = False, auto_update_seq_features = False):
        """Updates the model attributes
        This method is used to add/remove model features or data/generated
        sequences. These changes will be propagated through the class to update
        any other attributes that need to match (e.g. the marginals or
        seq_features).

        Parameters
        ----------
        add_data_seqs : list
            List of CDR3 sequences to add to data_seq pool.
        add_gen_seqs : list
            List of CDR3 sequences to add to data_seq pool.
        add_gen_seqs : list
            List of CDR3 sequences to add to data_seq pool.
        add_features : list
            List of feature lists to add to self.features
        remove_featurese : list
            List of feature lists and/or indices to remove from self.features
        add_constant_features : list
            List of feature lists to add to constant features. (Not currently used)
        auto_update_marginals : bool
            Specifies to update marginals.
        auto_update_seq_features : bool
            Specifies to update seq features.

        Attributes set
        --------------
        features : list
            List of model features
        data_seq_features : list
            Features data_seqs have been projected onto.
        gen_seq_features : list
            Features gen_seqs have been projected onto.
        data_marginals : ndarray
            Marginals over the data sequences for each model feature.
        gen_marginals : ndarray
            Marginals over the generated sequences for each model feature.
        model_marginals : ndarray
            Marginals over the generated sequences, reweighted by the model,
            for each model feature.

        """

        self.data_seqs += [[seq,'',''] if type(seq)==str else seq for seq in add_data_seqs] #add_data_seqs
        self.gen_seqs += [[seq,'',''] if type(seq)==str else seq for seq in add_gen_seqs] #add_gen_seqs

        if len(remove_features) > 0:
            indices_to_keep = [i for i, feature_lst in enumerate(self.features) if feature_lst not in remove_features and i not in remove_features]
            self.features = self.features[indices_to_keep]
            self.update_model_structure(initialize=True)
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}

        if len(add_features) > 0:
            if len(self.features) == 0:
                self.features = np.array(add_features, dtype=object)
            else:
                self.features = np.append(self.features, add_features, axis = 0)
            self.update_model_structure(initialize=True)
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}

        if (len(add_data_seqs + add_features + remove_features) > 0 or auto_update_seq_features) and len(self.features)>0 and len(self.data_seqs)>0:
            print('Encode data.')
            self.data_seq_features = [self.find_seq_features(seq) for seq in tqdm(self.data_seqs)]

        if (len(add_data_seqs + add_features + remove_features) > 0 or auto_update_marginals > 0) and len(self.features)>0:
            self.data_marginals = self.compute_marginals(seq_model_features = self.data_seq_features, use_flat_distribution = True)

        if (len(add_gen_seqs + add_features + remove_features) > 0 or auto_update_seq_features) and len(self.features)>0 and len(self.gen_seqs)>0:
            print('Encode gen.')
            self.gen_seq_features = [self.find_seq_features(seq) for seq in tqdm(self.gen_seqs)]


        if (len(add_gen_seqs + add_features + remove_features) > 0 or auto_update_marginals) and len(self.features)>0:
            self.gen_marginals = self.compute_marginals(seq_model_features = self.gen_seq_features, use_flat_distribution = True)
            self.model_marginals = self.compute_marginals(seq_model_features = self.gen_seq_features)

    def add_generated_seqs(self, num_gen_seqs = 0, reset_gen_seqs = True, custom_model_folder = None, add_error=False,custom_error=None):
        """Generates MonteCarlo sequences for gen_seqs using OLGA.

        Only generates seqs from a V(D)J model. Requires the OLGA package
        (pip install olga).

        Parameters
        ----------
        num_gen_seqs : int or float
            Number of MonteCarlo sequences to generate and add to the specified
            sequence pool.
        custom_model_folder : str
            Path to a folder specifying a custom IGoR formatted model to be
            used as a generative model. Folder must contain 'model_params.txt'
            and 'model_marginals.txt'
        add_error: bool
            simualate sequencing error: default is false
        custom_error: int
            set custom error rate for sequencing error.
            Default is the one inferred by igor.

        Attributes set
        --------------
        gen_seqs : list
            MonteCarlo sequences drawn from a VDJ recomb model
        gen_seq_features : list
            Features gen_seqs have been projected onto.

        """
        from sonia.utils import add_random_error
        from olga.utils import nt2aa

        #Load generative model
        if custom_model_folder is None:
            try:
                if self.custom_pgen_model is None: main_folder = os.path.join(os.path.dirname(__file__), 'default_models', self.chain_type)
                else: main_folder=self.custom_pgen_model
            except:
                main_folder = os.path.join(os.path.dirname(__file__), 'default_models', self.chain_type)
        else:
            main_folder = custom_model_folder

        params_file_name = os.path.join(main_folder,'model_params.txt')
        marginals_file_name = os.path.join(main_folder,'model_marginals.txt')
        V_anchor_pos_file = os.path.join(main_folder,'V_gene_CDR3_anchors.csv')
        J_anchor_pos_file = os.path.join(main_folder,'J_gene_CDR3_anchors.csv')

        if not os.path.isfile(params_file_name) or not os.path.isfile(marginals_file_name):
            print('Cannot find specified custom generative model files: ' + '\n' + params_file_name + '\n' + marginals_file_name)
            print('Exiting sequence generation...')
            return None
        if not os.path.isfile(V_anchor_pos_file):
            V_anchor_pos_file = os.path.join(os.path.dirname(olga_load_model.__file__), 'default_models', self.chain_type, 'V_gene_CDR3_anchors.csv')
        if not os.path.isfile(J_anchor_pos_file):
            J_anchor_pos_file = os.path.join(os.path.dirname(olga_load_model.__file__), 'default_models', self.chain_type, 'J_gene_CDR3_anchors.csv')

        with open(params_file_name,'r') as file:
            sep=0
            error_rate=''
            lines=file.read().splitlines()
            while len(error_rate)<1:
                error_rate=lines[-1+sep]
                sep-=1

        if custom_error is None: self.error_rate=float(error_rate)
        else: self.error_rate=custom_error

        if self.vj:
            genomic_data = olga_load_model.GenomicDataVJ()
            genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
            generative_model = olga_load_model.GenerativeModelVJ()
            generative_model.load_and_process_igor_model(marginals_file_name)
            sg_model = seq_gen.SequenceGenerationVJ(generative_model, genomic_data)
        else:
            genomic_data = olga_load_model.GenomicDataVDJ()
            genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
            generative_model = olga_load_model.GenerativeModelVDJ()
            generative_model.load_and_process_igor_model(marginals_file_name)
            sg_model = seq_gen.SequenceGenerationVDJ(generative_model, genomic_data)

        #Generate sequences
        print('Generate sequences.')
        if add_error: seqs = [[nt2aa(add_random_error(seq[0],self.error_rate)), genomic_data.genV[seq[2]][0].split('*')[0], genomic_data.genJ[seq[3]][0].split('*')[0]] for seq in [sg_model.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ') for _ in tqdm(range(int(num_gen_seqs)))]]
        else: seqs = [[seq[1], genomic_data.genV[seq[2]][0].split('*')[0], genomic_data.genJ[seq[3]][0].split('*')[0]] for seq in [sg_model.gen_rnd_prod_CDR3(conserved_J_residues='ABCEDFGHIJKLMNOPQRSTUVWXYZ') for _ in tqdm(range(int(num_gen_seqs)))]]
        if reset_gen_seqs: #reset gen_seqs if needed
            self.gen_seqs = []
        #Add to specified pool(s)
        self.update_model(add_gen_seqs = seqs)

    def save_model(self, save_dir, attributes_to_save = None,force=True):
        """Saves model parameters and sequences

        Parameters
        ----------
        save_dir : str
            Directory name to save model attributes to.

        attributes_to_save: list
            name of attributes to save

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
                L1_file.write('min_energy_clip ='+str(self.min_energy_clip)+'\n')
                L1_file.write('max_energy_clip ='+str(self.max_energy_clip)+'\n')
                L1_file.write('likelihood_train,likelihood_test\n')
                for i in range(len(self.likelihood_train)):
                    L1_file.write(str(self.likelihood_train[i])+','+str(self.likelihood_test[i])+'\n')

        if 'model' in attributes_to_save:
            with open(os.path.join(save_dir, 'features.tsv'), 'w') as feature_file:
                feature_file.write('Feature,marginal_data,marginal_model,marginal_gen\n')
                for i in range(len(self.features)):feature_file.write(';'.join(self.features[i])+','+str(self.data_marginals[i])+','+str(self.model_marginals[i])+','+str(self.gen_marginals[i])+'\n')
            self.model.save(os.path.join(save_dir, 'model.h5'))
        
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

    def load_model(self, load_dir = None, load_seqs = True, feature_file = None, model_file = None, data_seq_file = None, gen_seq_file = None, log_file = None, verbose = True):
        """Loads model from directory.

        Parameters
        ----------
        load_dir : str
            Directory name to load model attributes from.

        """

        if load_dir is not None:
            if not os.path.isdir(load_dir):
                print('Directory for loading model does not exist (' + load_dir + ')')
                print('Exiting...')
                return None
            if feature_file is None: feature_file = os.path.join(load_dir, 'features.tsv')
            if model_file is None: model_file = os.path.join(load_dir, 'model.h5')
            if data_seq_file is None: data_seq_file = os.path.join(load_dir, 'data_seqs.tsv')
            if gen_seq_file is None: gen_seq_file = os.path.join(load_dir, 'gen_seqs.tsv')
            if log_file is None: log_file = os.path.join(load_dir, 'log.txt')


        self._load_features_and_model(feature_file, model_file, verbose)

        if data_seq_file is None:
            pass
        elif os.path.isfile(data_seq_file) and load_seqs:
            with open(data_seq_file, 'r') as data_seqs_file:
                self.data_seqs = []
                self.data_seq_features = []
                for line in data_seqs_file.read().strip().split('\n')[1:]:
                    split_line = line.split('\t')
                    self.data_seqs.append(split_line[0].split(';'))
                    self.data_seq_features.append([self.feature_dict[tuple(f.split(','))] for f in split_line[2].split(';') if tuple(f.split(',')) in self.feature_dict])
        elif load_seqs and verbose:
            print('Cannot find data_seqs.tsv  --  no data seqs loaded.')


        if gen_seq_file is None:
            pass
        elif os.path.isfile(gen_seq_file) and load_seqs:
            with open(gen_seq_file, 'r') as gen_seqs_file:
                self.gen_seqs = []
                self.gen_seq_features = []
                for line in gen_seqs_file.read().strip().split('\n')[1:]:
                    split_line = line.split('\t')
                    self.gen_seqs.append(split_line[0].split(';'))
                    self.gen_seq_features.append([self.feature_dict[tuple(f.split(','))] for f in split_line[2].split(';') if tuple(f.split(',')) in self.feature_dict])
        elif load_seqs and verbose:
            print('Cannot find gen_seqs.tsv  --  no generated seqs loaded.')

        #self.update_model(auto_update_marginals = True) # to check

        if log_file is None:
            self.L1_converge_history = []
        elif os.path.isfile(log_file):
            with open(log_file, 'r') as L1_file:
                self.L1_converge_history=[]
                for i,line in enumerate(L1_file):
                    if i==0: self.Z=float(line.strip().split('=')[1])
                    elif i==1: self.norm_productive=float(line.strip().split('=')[1])
                    elif i==2: 
                        try: 
                            self.min_energy_clip=float(line.strip().split('=')[1])
                        except:
                            continue
                    elif i==3: 
                        try:
                            self.max_energy_clip=float(line.strip().split('=')[1])
                        except:
                            continue
                    elif len(line.strip())>0 and i>4: 
                        try:
                            self.likelihood_train.append(float(line.strip().split(',')[0]))
                            self.likelihood_test.append(float(line.strip().split(',')[1]))
                        except:
                            continue

        else:
            self.L1_converge_history = []
            self.likelihood_train= []
            self.likelihood_test= []
            if verbose: print('Cannot find log.txt  --  no norms and convergence loaded.')

        return None

    def _load_features_and_model(self, feature_file, model_file, verbose = True):
        """Loads features and model.

        This is set as an internal function to allow daughter classes to load
        models from saved feature energies directly.
        """


        if feature_file is None and verbose:
            print('No feature file provided --  no features loaded.')
        elif os.path.isfile(feature_file):
            features = []
            data_marginals=[]
            gen_marginals=[]
            model_marginals=[]
            with open(feature_file, 'r') as features_file:
                all_lines = features_file.read().strip().split('\n')[1:] #skip header
                splitted=[l.split(',') for l in all_lines]
                features = np.array([l[0].split(';') for l in splitted])
                data_marginals=[float(l[1]) for l in splitted]
                model_marginals=[float(l[2]) for l in splitted]
                gen_marginals=[float(l[3]) for l in splitted]
            
            self.features = np.array(features)
            self.feature_dict = {tuple(f): i for i, f in enumerate(self.features)}
            self.data_marginals=data_marginals
            self.model_marginals=model_marginals
            self.gen_marginals=gen_marginals

        elif verbose:
            print('Cannot find features file or model file --  no features loaded.')

        if model_file is None and verbose:
            print('No model file provided -- no model parameters loaded.')
        elif os.path.isfile(model_file):
            self.model = load_model(model_file, custom_objects={'loss': self._loss,'likelihood': self._likelihood}, compile = False)
            self.optimizer = RMSprop()
            self.model.compile(optimizer=self.optimizer, loss=self._loss,metrics=[self._likelihood])
        elif verbose:
            print('Cannot find model file --  no model parameters loaded.')
