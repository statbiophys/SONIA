# Directory architecture
```
SONIA/
├── LICENSE
├── MANIFEST.in
├── README.md
├── data_infer.txt
├── data_seqs.txt
├── gen_seqs.txt
├── setup.py
└── sonia
    ├── __init__.py
    ├── evaluate.py
    ├── evaluate_model.py
    ├── generate.py
    ├── infer.py
    ├── plotting.py
    ├── sequence_generation.py
    ├── sonia.py
    ├── sonia_leftpos_rightpos.py
    ├── sonia_length_pos.py
    ├── utils.py
    └── default_models
        ├── human_B_heavy
        │   ├── J_gene_CDR3_anchors.csv
        │   ├── V_gene_CDR3_anchors.csv
        │   ├── features.tsv
        │   ├── log.txt
        │   ├── model_marginals.txt
        │   └── model_params.txt
        ├── human_B_kappa
        │   ├── J_gene_CDR3_anchors.csv
        │   ├── V_gene_CDR3_anchors.csv
        │   ├── features.tsv
        │   ├── log.txt
        │   ├── model_marginals.txt
        │   └── model_params.txt
        ├── human_B_lambda
        │   ├── J_gene_CDR3_anchors.csv
        │   ├── V_gene_CDR3_anchors.csv
        │   ├── features.tsv
        │   ├── log.txt
        │   ├── model_marginals.txt
        │   └── model_params.txt
        ├── human_T_alpha
        │   ├── J_gene_CDR3_anchors.csv
        │   ├── V_gene_CDR3_anchors.csv
        │   ├── features.tsv
        │   ├── log.txt
        │   ├── model_marginals.txt
        │   └── model_params.txt
        ├── human_T_beta
        │   ├── J_gene_CDR3_anchors.csv
        │   ├── V_gene_CDR3_anchors.csv
        │   ├── features.tsv
        │   ├── log.txt
        │   ├── model_marginals.txt
        │   └── model_params.txt
        └── mouse_T_beta
            ├── J_gene_CDR3_anchors.csv
            ├── V_gene_CDR3_anchors.csv
            ├── features.tsv
            ├── log.txt
            ├── model_marginals.txt
            └── model_params.txt

```
## Notes about training data preparation

Sonia shines when trained on top of independent rearrangement events, thus you should throw away the read count information.
If you have a sample from an individual, you should keep the unique nucleotide rearrangements. This means that in principle there could be few aminoacid CDR3,V,J combination that are not unique after the mapping from nucleotide to aminoacid, but that's fine. Moreover if you pool data from multiple people you can still keep rearrangements that are found in multiple individuals because you are sure that they correspond to independent recombination events.

## Notes about CDR3 sequence definition

This code is quite flexible, however it does demand a very consistent definition of CDR3 sequences.

**CHECK THE DEFINITION OF THE CDR3 REGION OF THE SEQUENCES YOU INPUT.** This will likely be the most often problem that occurs.

The default models/genomic data are set up to define the CDR3 region from the conserved cysteine C (INCLUSIVE) in the V region to the conserved F or W (INCLUSIVE) in the J. This corresponds to positions X and X according to IMGT. This can be changed by altering the anchor position files, however the user is strongly recommended against this.