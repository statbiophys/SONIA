## Sonia modules in a Python script
In order to incorporate the core algorithm into an analysis pipeline (or to write your own script wrappers) all that is needed is to import the modules. Each module defines some classes that only a few methods get called on.

The modules are:

| Module name                                    | Classes                                          |    
|------------------------------------------------|--------------------------------------------------|
| evaluate_model.py                              | EvaluateModel                                    |
| sequence_generation.py                         | SequenceGeneration                               |
| plotting.py.                                   | Plotter                                          |
| sonia_leftpos_rightpos.py                      | SoniaLeftposRightpos                             |
| sonia_length_pos.py                            | SoniaLengthPos                                   |
| sonia_vjl.py                                   | SoniaVJL                                         |
| sonia.py                                       | Sonia                                            |
| utils.py                                       | N/A (contains util functions)                    |

The classes with methods that are of interest will be EvaluateModel (to evaluate seqs) and SequenceGeneration (to generate seqs), SoniaLeftposRightpos or SoniaLengthPos (to initialise and infer the models) and Plotter (to plot results).

You can find some examples in the examples folder. We demonstrate here some basic usage.
Data and gen files are included in the GitHub repository to demonstrate usage, however we recommend to expand both data and generated files for an accurate inference.

```
import os
import sonia
from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
from sonia.plotting import Plotter
from sonia.evaluate_model import EvaluateModel
from sonia.sequence_generation import SequenceGeneration

work_folder = 'examples/' # where data files are and output folder should be
data_file = work_folder + 'data_seqs.txt' # file with data sequences
gen_file = work_folder + 'gen_seqs.txt' # file with generated sequences if not generated internally
output_folder = work_folder + 'selection/' # location to save model

# load lists of sequences with gene specification
with open(data_file) as f: # this assume data sequences are in semi-colon separated text file, with gene specification
    data_seqs = [x.strip().split(';') for x in f]

gen_seqs = []
with open(gen_file) as f:  # this assume generated sequences are in semi-colon separated text file, with gene specification
    gen_seqs = [x.strip().split(';') for x in f]

# creates the model object, load up sequences and set the features to learn
qm = SoniaLeftposRightpos(data_seqs=data_seqs, gen_seqs=gen_seqs)

# infer model
qm.infer_selection()

# plot results
pl=Plotter(qm)
pl.plot_model_learning('model_learning.png')
pl.plot_vjl('marginals.png')
pl.plot_logQ('log_Q.png')

# save the model
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
qm.save_model(output_folder + 'SONIA_model_example')

# load default model (human TRA)
model_dir=os.path.join(os.path.dirname(sonia.sonia_leftpos_rightpos.__file__),'default_models','human_T_alpha')
qm=SoniaLeftposRightpos(load_dir=model_dir,chain_type='human_T_alpha')

# load evaluation and generation classes
ev=EvaluateModel(sonia_model=qm)
sq=SequenceGeneration(sonia_model=qm)

# generate seqs pre
seqs_pre=sq.generate_sequences_pre(10)

# generate seqs post
seqs_post = sq.generate_sequences_post(10)
print(seqs_post)

# evaluate Q, pgen and ppost of sequences 
# NB: data has to be in format: list(array((n_seqs,3 or more))). Check output of generate_sequences_post method for an example (4th column is not used in the evaluate_seqs method).
qs,pgens,pposts= ev.evaluate_seqs(seqs_post)
print(pgens,pposts,qs)
```

Additional documentation of the modules is found in their docstrings (accessible either through pydoc or using help() within the python interpreter).