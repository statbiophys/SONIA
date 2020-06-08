## Command line console scripts

There are three command line console scripts (the scripts can still be called as executables if SONIA is not installed):
1. ```sonia-evaluate```
  * evaluates Ppost, Pgen or selection factors of sequences according to a generative V(D)J model and selection model.
2. ```sonia-generate```
  * generates CDR3 sequences, before (like olga) or after selection
3. ```sonia-infer```
  * infers a selection model with respect to a generative V(D)J model

For any of them you can execute with the -h or --help flags to get the options.

### Quick Demo
After installing SONIA, we offer a quick demonstration of the console scripts. This will demonstrate generating and evaluating sequences and infer a selection model from the default model for human TCR beta chains that ships with SONIA. 

1. ```$ sonia-evaluate --humanTRB CASSTGNYGAFF --v_mask TRBV9 --j_mask TRBJ1-1 --ppost```
  * This computes Ppost,Pgen and Q of the TCR CASSTGNYGAFF,TRBV9,TRBJ1-1 (you should get ~1.3e-11, ~9.2e-12 and ~1.4 respectively)

2. ```$ sonia-generate --humanTRB -n 5 --pre```
  * Generate 5 human TRB CDR3 sequences from the pre-selection repertoire and print to stdout along with the V and J genes used to generate them.

3. ```$ sonia-generate --humanTRB -n 10000 --post -o example_seqs.txt```
  * This generates a file example_seqs.tsv and writes 10000 generated human TRB CDR3 sequences from the post-selection repertoire.

4. ```$ sonia-evaluate --humanTRB --ppost -i example_seqs.txt -m 5  -o example_evaluation.txt```
  * This reads in the first 5 sequences from the file we just generated, example_seqs.tsv, evaluates them and writes the results them to the file example_pgens.tsv
  
5. ```$ sonia-infer --humanTRB -i example_seqs.txt -o sel_model```
  * This reads in the full file example_seqs.txt, infers a selection model and saves to the folder sel_model
  
### Specifying a default V(D)J model (or a custom model folder)
All of the console scripts require specifying a V(D)J model. SONIA ships with 6 default models that can be indicated by flags, or a custom model folder can be indicated.

| Options                                         | Description                                      |
|-------------------------------------------------|--------------------------------------------------|
| **--humanTRA**                                  | Default human T cell alpha chain model (VJ)      |
| **--humanTRB**                                  | Default human T cell beta chain model (VDJ)      |
| **--mouseTRB**                                  | Default mouse T cell beta chain model (VDJ)      |
| **--humanIGH**                                  | Default human B cell heavy chain model (VDJ)     |
| **--humanIGK**                                  | Default human B cell light kappa chain model (VJ)|
| **--humanIGL**                                  | Default human B cell light lambda chain model (VJ)|
| **--set_custom_model_VJ** PATH/TO/MODEL_FOLDER/ | Specifies the directory PATH/TO/MODEL_FOLDER/ of a custom VJ generative model|
| **--set_custom_model_VDJ** PATH/TO/MODEL_FOLDER/| Specifies the directory PATH/TO/MODEL_FOLDER/ of a custom VDJ generative model|

Note, if specifying a folder for a custom VJ recombination model
(e.g. an alpha or light chain model) or a custom VDJ recombination model
(e.g. a beta or heavy chain model), the folder must contain the following files
with the exact naming convention:

* model_params.txt 
* model_marginals.txt 
* V_gene_CDR3_anchors.csv (V anchor residue position and functionality file)
* J_gene_CDR3_anchors.csv (J anchor residue position and functionality file)
* features.tsv (if you want to load the selection model as well: not required for in the sonia-infer command)
* log.txt (if you want to load the selection model as well: not required in the sonia-infer command) 

The console scripts can only read files of the assumed anchor.csv/[IGoR](https://github.com/qmarcou/IGoR) syntaxes. See the default models in the sonia directory for examples.

### Options common to all commands

| Options                                         | Description                                      |
|-------------------------------------------------|--------------------------------------------------|
|  -h, --help                                     |     show command options                         |
|  --sonia_model=MODEL_TYPE                       | specify model type: leftright or lengthpos. Default is leftright |
|  -i PATH/TO/FILE, --infile=PATH/TO/FILE         | read in CDR3 sequences (and optionally V/J masks) from PATH/TO/FILE|
|  -o PATH/TO/FILE, --outfile=PATH/TO/FILE        | output to PATH/TO/FILE                           |
|  --seq_in=INDEX, --seq_index=INDEX              | specifies sequences to be read in are in column INDEX. Default is index 0 (the first column). |
|  --v_in=INDEX, --v_mask_index=INDEX             | specifies V_masks are found in column INDEX in the input file. Default is 1. |
|  --j_in=INDEX, --j_mask_index=INDEX             | specifies J_masks are found in column INDEX in the input file. Default is 2. |
|  -m N, --max_number_of_seqs=N                   | read at most N sequences.                        |
|  --lines_to_skip=N                              | skip the first N lines of the file. Default is 0.|
|  -d DELIMITER, --delimiter=DELIMITER            | declare infile delimiter. Default is tab for .tsv input files, comma for .csv files, and any whitespace for all others. Choices: 'tab', 'space', ',', ';', ':' |
|  --delimiter_out=DELIMITER_OUT                  | declare outfile delimiter. Default is tab for .tsv output files, comma for .csv files, and the infile delimiter for all others. Choices: 'tab', 'space', ',', ';', ':' |

for command specific options use the --help flag.