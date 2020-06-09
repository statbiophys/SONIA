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