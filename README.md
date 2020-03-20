# SONIA

## Synopsis

SONIA is a python 2.7/3.6  software developed to infer selection pressures on features of amino acid CDR3 sequences. The inference is based on maximizing the likelihood of observing a selected data sample given a representative pre-selected sample. This method was first used in Elhanati et al (2014) to study thymic selection. For this purpose, the pre-selected sample can be generated internally using the OLGA software package, but SONIA allows it also to be supplied externally, in the same way the data sample is provided.

SONIA takes as input TCR CDR3 amino acid sequences, with or without per sequence lists of possible V and J genes suspected to be used in the recombination process for this sequence. As in Elhanati (2014), its output is selection factors for each amino acid / position / CDR3 length combinations, and also for each V and J gene choice. These selection factors can be used to calculate sequence level selection factors, or energies (log of selection factors), which indicate how more or less represented this sequence would be in the selected pool as compared to the the pre-selected pool. These in turn could be used to calculate the probability to observe any sequence after selection. A convenience class EvaluateModel is included that can load a previously inferred model and perform such tasks.

An example script is provided that reads in selected and pre-selected sequences from supplied text files and infer selection factors on any amino acid / position / CDR3 length combinations and V/J identity, saving the inferred model to a file. Then the model is loaded into the EvaluateModel to generate sequences before and after selection, and calculate probabilities and energies for the generated sequences.

Free use of SONIA is granted under the terms of the GNU General Public License version 3 (GPLv3).

## Version
Latest released version: 0.0.1

## Installation
OLGA is a python 2.7/3.6 software. It is available on PyPI and can be downloaded and installed through pip: ```pip install olga```.

OLGA is also available on [GitHub](https://github.com/statbiophys/SONIA). The command line entry points can be installed by using the setup.py script: ```$ python setup.py install```.

Directory architecture:
```
SONIA/
├── LICENSE
├── MANIFEST.in
├── README.md
├── data_seqs.txt
├── example_pipeline_script.py
├── gen_seqs.txt
├── setup.py
└── sonia
    ├── __init__.py
    ├── default_models
    │   ├── human_B_heavy
    │   │   ├── J_gene_CDR3_anchors.csv
    │   │   ├── V_gene_CDR3_anchors.csv
    │   │   ├── features.tsv
    │   │   ├── log.txt
    │   │   ├── model_marginals.txt
    │   │   └── model_params.txt
    │   ├── human_T_alpha
    │   │   ├── J_gene_CDR3_anchors.csv
    │   │   ├── V_gene_CDR3_anchors.csv
    │   │   ├── features.tsv
    │   │   ├── log.txt
    │   │   ├── model_marginals.txt
    │   │   └── model_params.txt
    │   ├── human_T_beta
    │   │   ├── J_gene_CDR3_anchors.csv
    │   │   ├── V_gene_CDR3_anchors.csv
    │   │   ├── features.tsv
    │   │   ├── log.txt
    │   │   ├── model_marginals.txt
    │   │   └── model_params.txt
    │   └── mouse_T_beta
    │       ├── J_gene_CDR3_anchors.csv
    │       ├── V_gene_CDR3_anchors.csv
    │       ├── features.tsv
    │       ├── log.txt
    │       ├── model_marginals.txt
    │       └── model_params.txt
    ├── evaluate.py
    ├── evaluate_model.py
    ├── generate.py
    ├── infer.py
    ├── plotting.py
    ├── sequence_generation.py
    ├── sonia.py
    ├── sonia_leftpos_rightpos.py
    ├── sonia_length_pos.py
    └── utils.py
```

## Command line console scripts and Examples

There are two command line console scripts (the scripts can still be called as executables if OLGA is not installed):
1. sonia-evaluate
  * evaluates Ppost, Pgen or selection factors of sequences according to a generative V(D)J model and selection model Q
2. sonia-generate
  * generates CDR3 sequences from before or after selection
3. sonia-infer
  * infers a selection model with respect to a generative V(D)J model

For any of them you can execute with the -h or --help flags to get the options.

## Contact

Any issues or questions should be addressed to [us](mailto:zachary.sethna@gmail.com).

## License

Free use of SONIA is granted under the terms of the GNU General Public License version 3 (GPLv3).