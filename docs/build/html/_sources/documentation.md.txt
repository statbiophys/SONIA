# Documentation

## sonia.sonia

Created on Wed Jan 30 12:06:58 2019

@author: zacharysethna and Giulio Isacchini

### Classes
`Sonia(features=[], data_seqs=[], gen_seqs=[], chain_type='humanTRB',
load_dir=None, feature_file=None, model_file=None, data_seq_file=None,
gen_seq_file=None, log_file=None, load_seqs=True, l2_reg=0.0, min_energy_clip=-5,
max_energy_clip=10, seed=None, vj=False)`

    Class used to infer a Q selection model.
    
    
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

#### Methods

`add_generated_seqs(self, num_gen_seqs=0, reset_gen_seqs=True, custom_model_folder=None, add_error=False, custom_error=None)`

        Generates MonteCarlo sequences for gen_seqs using OLGA.
        
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

`compute_energy(self, seqs_features)`

        Computes the energy of a list of sequences according to the model.
        
        Parameters
        ----------
        seqs_features : list
            list of encoded sequences into sonia features.
        
        Returns
        -------
        E : float
            Energies of seqs according to the model.

`compute_marginals(self, features=None, seq_model_features=None, seqs=None, use_flat_distribution=False, output_dict=False)`
        
        Computes the marginals of each feature over sequences.

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

`compute_seq_energy(self, seq=None, seq_features=None)`

        Computes the energy of a sequence according to the model.
        
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

`find_seq_features(self, seq, features=None)`

        Finds which features match seq
        
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

`infer_selection(self, epochs=10, batch_size=5000, initialize=True, seed=None, validation_split=0.2, monitor=False, verbose=0)`

        Infer model parameters, i.e. energies for each model feature.
        
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

`load_model(self, load_dir=None, load_seqs=True, feature_file=None, model_file=None, data_seq_file=None, gen_seq_file=None, log_file=None, verbose=True)`

        Loads model from directory.
        
        Parameters
        ----------
        load_dir : str
            Directory name to load model attributes from.

`save_model(self, save_dir, attributes_to_save=None, force=True)`

        Saves model parameters and sequences
        
        Parameters
        ----------
        save_dir : str
            Directory name to save model attributes to.
        
        attributes_to_save: list
            name of attributes to save

`seq_feature_proj(self, feature, seq)`

        Checks if a sequence matches all subfeatures of the feature list
        
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

`update_model(self, add_data_seqs=[], add_gen_seqs=[], add_features=[], remove_features=[], add_constant_features=[], auto_update_marginals=False, auto_update_seq_features=False)`
        
        Updates the model attributes
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

`update_model_structure(self, output_layer=[], input_layer=[], initialize=False)`

        Defines the model structure and compiles it.
        
        Parameters
        ----------
        structure : Sequential Model Keras
            structure of the model
        
        initialize: bool
            if True, it initializes to linear model, otherwise it updates to new structure
            
## sonia.sonia_leftpos_rightpos

@author: zacharysethna

### Classes


`SoniaLeftposRightpos(data_seqs=[], gen_seqs=[], chain_type='humanTRB', load_dir=None, feature_file=None, data_seq_file=None, gen_seq_file=None, log_file=None, load_seqs=True, max_depth=25, max_L=30, include_indep_genes=False, include_joint_genes=True, min_energy_clip=-5, max_energy_clip=10, seed=None, custom_pgen_model=None, l2_reg=0.0, vj=False)`
    
    Class used to infer a Q selection model.
    
    
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

#### Methods

`add_features(self, include_indep_genes=False, include_joint_genes=True, custom_pgen_model=None)`
        
        Generates a list of feature_lsts for L/R pos model.
        
        Parameters
        ----------
        include_genes : bool
            If true, features for gene selection are also generated. Currently
            joint V/J pairs used.
        
        custom_pgen_model: string
            path to folder of custom olga model.

`compute_seq_energy_from_parameters(self, seqs=None, seqs_features=None)`
        
        Computes the energy of a list of sequences according to the model.
        
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

`find_seq_features(self, seq, features=None)`
        
        Finds which features match seq
        
        
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

`get_energy_parameters(self, return_as_dict=False)`
        
        Extract energy terms from keras model.

`save_model(self, save_dir, attributes_to_save=None, force=True)`
        
        Saves model parameters and sequences
        
        Parameters
        ----------
        save_dir : str
            Directory name to save model attributes to.
        
        attributes_to_save: list
            Names of attributes to save
            
            
## sonia.sonia_length_pos

Created on Wed Mar  6 15:12:15 2019

@author: Zachary Sethna

### Classes


`SoniaLengthPos(data_seqs=[], gen_seqs=[], chain_type='humanTRB', load_dir=None, feature_file=None, data_seq_file=None, gen_seq_file=None, log_file=None, min_L=4, max_L=30, include_indep_genes=False, include_joint_genes=True, min_energy_clip=-5, max_energy_clip=10, seed=None, custom_pgen_model=None, l2_reg=0.0, vj=False)`

    Class used to infer a Q selection model.
    
    
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

#### Methods

`add_features(self, include_indep_genes=False, include_joint_genes=True, custom_pgen_model=None)`
        
        Generates a list of feature_lsts for a length dependent L pos model.
        
        Parameters
        ----------
        include_genes : bool
            If true, features for gene selection are also generated. Currently
            joint V/J pairs used.
        
        custom_pgen_model: string
            path to folder of custom olga model.

`compute_seq_energy_from_parameters(self, seqs=None, seqs_features=None)`

        Computes the energy of a list of sequences according to the model.
        
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

`find_seq_features(self, seq, features=None)`

        Finds which features match seq
        
        If no features are provided, the length dependent amino acid model
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

`get_energy_parameters(self, return_as_dict=False)`

        Extract energy terms from keras model and gauge.
        
        For the length dependent position model, the gauge is set so that at a
        given position, for a given length, we have:
        
        <q_i,aa;L>_gen|L = 1
        
        
        Parameters
        ----------
        min_L : int
            Minimum length CDR3 sequence, if not given taken from class attribute
        max_L : int
            Maximum length CDR3 sequence, if not given taken from class attribute

`save_model(self, save_dir, attributes_to_save=None, force=True)`

        Saves model parameters and sequences
        
        Parameters
        ----------
        save_dir : str
            Directory name to save model attributes to.
        
        attributes_to_save: list
            Names of attributes to save
            
## sonia.sonia_vjl

@author: Giulio Isacchini

### Classes


`SoniaVJL(data_seqs=[], gen_seqs=[], chain_type='humanTRB', load_dir=None, feature_file=None, data_seq_file=None, gen_seq_file=None, log_file=None, load_seqs=True, max_depth=25, max_L=30, include_indep_genes=False, include_joint_genes=True, min_energy_clip=-5, max_energy_clip=10, seed=None, custom_pgen_model=None, l2_reg=0.0, vj=False, joint_vjl=False)`
    
    Class used to infer a Q selection model.
    
    
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

#### Methods

`add_features(self, custom_pgen_model=None)`

        Generates a list of feature_lsts for L/R pos model.
        
        Parameters
        ----------
        include_genes : bool
            If true, features for gene selection are also generated. Currently
            joint V/J pairs used.
        
        custom_pgen_model: string
            path to folder of custom olga model.

`compute_seq_energy_from_parameters(self, seqs=None, seqs_features=None)`

        Computes the energy of a list of sequences according to the model.
        
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

`find_seq_features(self, seq, features=None)`

        Finds which features match seq
        
        
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

`get_energy_parameters(self, return_as_dict=False)`

        Extract energy terms from keras model.

`save_model(self, save_dir, attributes_to_save=None)`

        Saves model parameters and sequences
        
        Parameters
        ----------
        save_dir : str
            Directory name to save model attributes to.
        
        attributes_to_save: list
            Names of attributes to save
            
## sonia.evaluate_model

@author: Giulio Isacchini

### Classes

`EvaluateModel(sonia_model=None, include_genes=True, processes=None, custom_olga_model=None)`
    
    Class used to evaluate sequences with the sonia model: Ppost=Q*Pgen
    
    
    Attributes
    ----------
    sonia_model: object
        Sonia model. Loaded previously, do not put the path.
    
    include_genes: bool
        Conditioning on gene usage for pgen/ppost evaluation. Default: True
    
    processes: int
        Number of processes to use to infer pgen. Default: all.
    
    custom_olga_model: object
        Optional: already loaded custom generation_probability olga model.
    
    Methods
    ----------
    
    evaluate_seqs(seqs=[])
        Returns Q, pgen and ppost of a list of sequences.
    
    evaluate_selection_factors(seqs=[])
        Returns normalised selection factor Q (Ppost=Q*Pgen) of a list of sequences (faster than evaluate_seqs because it does not compute pgen and ppost)

#### Methods

`compute_all_pgens(self, seqs)`

        Compute Pgen of sequences using OLGA in parallel
        
        Parameters
        ----------
        seqs: list
            list of sequences to evaluate.
        
        Returns
        -------
        pgens: array
            generation probabilities of the sequences.

`compute_joint_marginals(self)`

        Computes joint marginals for all.
        
        Attributes Set
        -------
        gen_marginals_two: array
            matrix (i,j) of joint marginals for pre-selection distribution
        data_marginals_two: array
            matrix (i,j) of joint marginals for data
        model_marginals_two: array
            matrix (i,j) of joint marginals for post-selection distribution
        gen_marginals_two_independent: array
            matrix (i,j) of independent joint marginals for pre-selection distribution
        data_marginals_two_independent: array
            matrix (i,j) of joint marginals for pre-selection distribution
        model_marginals_two_independent: array
            matrix (i,j) of joint marginals for pre-selection distribution

`evaluate_selection_factors(self, seqs=[])`

        Returns normalised selection factor Q (of Ppost=Q*Pgen) of list of sequences (faster than evaluate_seqs because it does not compute pgen and ppost)
        
        Parameters
        ----------
        seqs: list
            list of sequences to evaluate
        
        Returns
        -------
        Q: array
            selection factor Q (of Ppost=Q*Pgen) of the sequences

`evaluate_seqs(self, seqs=[])`

        Returns selection factors, pgen and pposts of sequences.
        
        Parameters
        ----------
        seqs: list
            list of sequences to evaluate
        
        Returns
        -------
        Q: array
            selection factor Q (of Ppost=Q*Pgen) of the sequences
        
        pgens: array
            pgen of the sequences
        
        pposts: array
            ppost of the sequences

`joint_marginals(self, features=None, seq_model_features=None, seqs=None, use_flat_distribution=False)`

        Returns joint marginals P(i,j) with i and j features of sonia (l3, aA6, etc..), index of features attribute is preserved.
        Matrix is upper-triangular.
        
        Parameters
        ----------
        features: list
            custom feature list 
        seq_model_features: list
            encoded sequences
        seqs: list
            seqs to encode.
        use_flat_distribution: bool
            for data and generated seqs is True, for model is False (weights with Q)
        
        Returns
        -------
        joint_marginals: array
            matrix (i,j) of joint marginals

`joint_marginals_independent(self, marginals)`

        Returns independent joint marginals P(i,j)=P(i)*P(j) with i and j features of sonia (l3, aA6, etc..), index of features attribute is preserved.
        Matrix is upper-triangular.
        
        Parameters
        ----------
        marginals: list
            marginals.
        
        Returns
        -------
        joint_marginals: array
            matrix (i,j) of joint marginals
            
## sonia.sequence_generation

@author: Giulio Isacchini

### Classes


`SequenceGeneration(sonia_model=None, custom_olga_model=None, custom_genomic_data=None)`
    
    Class used to evaluate sequences with the sonia model
    
    Attributes
    ----------
    sonia_model: object 
        Required. Sonia model: only object accepted.
    
    custom_olga_model: object 
        Optional: already loaded custom olga sequence_generation object
    
    custom_genomic_data: object
        Optional: already loaded custom olga genomic_data object
    
    Methods
    ----------
    
    generate_sequences_pre(num_seqs = 1)
        Generate sequences using olga
    
    generate_sequences_post(num_seqs,upper_bound=10)
        Generate sequences using olga and perform rejection selection.
    
    rejection_sampling(upper_bound=10,energies=None)
        Returns acceptance from rejection sampling of a list of energies.
        By default uses the generated sequences within the sonia model.

#### Methods

`generate_sequences_post(self, num_seqs=1, upper_bound=10, nucleotide=True)`

        Generates MonteCarlo sequences from Sonia through rejection sampling.
        
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

`generate_sequences_pre(self, num_seqs=1, nucleotide=True)`

        Generates MonteCarlo sequences for gen_seqs using OLGA.
        
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

`rejection_sampling(self, upper_bound=10, energies=None)`

        Returns acceptance from rejection sampling of a list of seqs.
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

## sonia.plotting

@author: Giulio Isacchini

### Classes

`Plotter(sonia_model=None)`
    
    Class used to do plotting
    
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
    
    plot_vjl(save_name = None)
        Plots marginals of V gene, J gene and cdr3 length
    
    plot_logQ(save_name=None)
        Plots logQ of data and generated sequences
    
    plot_ratioQ(self,save_name=None)
        Plots the ratio of P(Q) in data and pre-selected pool. Useful for model validation.

#### Methods

`norm_marginals(self, marg, min_L=None, max_L=None)`

        renormalizing the marginals accourding to length, so the sum of the marginals over all amino acid 
        for one position/length combination will be 1 (and not the fraction of CDR3s of this length)
            
        Parameters
        ----------
        marg : ndarray
            the marginal to renormalize
        min_L : int
            Minimum length CDR3 sequence, if not given taken from class attribute
        max_L : int
            Maximum length CDR3 sequence, if not given taken from class attribute

`plot_logQ(self, save_name=None)`

        Plots logQ of data and generated sequences
        
        Parameters
        ----------
        save_name : str or None
            File name to save output figure. If None (default) does not save.

`plot_marginals_length_corrected(self, min_L=8, max_L=16, log_scale=True)`

        plot length normalized marginals using plot_onepoint_values
        
        Parameters
        ----------
        min_L : int
            Minimum length CDR3 sequence, if not given taken from class attribute
        max_L : int
            Maximum length CDR3 sequence, if not given taken from class attribute
        log_scale : bool
            if True (default) plots marginals on a log scale

`plot_model_learning(self, save_name=None)`

        Plots L1 convergence curve and marginal scatter.
        
        Parameters
        ----------
        save_name : str or None
            File name to save output figure. If None (default) does not save.

`plot_model_parameters(self, low_freq_mask=0.0)`

        plot the model parameters using plot_onepoint_values
        
        Parameters
        ----------
        low_freq_mask : float
            threshold on the marginals, anything lower would be grayed out

`plot_onepoint_values(self, onepoint=None, onepoint_dict=None, min_L=None, max_L=None, min_val=None, max_value=None, title='', cmap='seismic', bad_color='black', aa_color='white', marginals=False)`

        plot a function of aa, length and position from left, one heatplot per aa
           
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

`plot_prob(self, data=[], gen=[], model=[], n_bins=30, save_name=None, bin_min=-20, bin_max=-5, ptype='P_{pre}', figsize=(6, 4))`

        Histogram plot of Pgen/ppost/Q
        
        Parameters
        ----------
        n_bins: int
            number of bins of the histogram

`plot_ratioQ(self, save_name=None)`

        Plots the ratio of P(Q) in data and pre-selected pool. Useful for model validation. 
        
        Parameters
        ----------
        save_name : str or None
            File name to save output figure. If None (default) does not save.

`plot_vjl(self, save_name=None)`

        Plots marginals of V gene, J gene and cdr3 length
        
        Parameters
        ----------
        save_name : str or None
            File name to save output figure. If None (default) does not save.