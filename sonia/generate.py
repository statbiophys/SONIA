#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command line script to generate sequences.

    Copyright (C) 2020 Isacchini Giulio

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

This program generates sequences
"""

from __future__ import print_function, division,absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from optparse import OptionParser
import olga.sequence_generation as sequence_generation
from sonia.sonia_length_pos import SoniaLengthPos
from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
from sonia.evaluate_model import EvaluateModel
from sonia.sequence_generation import SequenceGeneration
import olga.load_model as olga_load_model
import numpy as np
from tqdm import tqdm

#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass

def chuncks(n,size):
    if n%size: return int(n/size)*[size]+[n%size]
    else : return int(n/size)*[size]

def main():
    """ Generate sequences."""

    parser = OptionParser(conflict_handler="resolve")

    #specify model
    parser.add_option('--humanTRA', '--human_T_alpha', action='store_true', dest='humanTRA', default=False, help='use default human TRA model (T cell alpha chain)')
    parser.add_option('--humanTRB', '--human_T_beta', action='store_true', dest='humanTRB', default=False, help='use default human TRB model (T cell beta chain)')
    parser.add_option('--mouseTRB', '--mouse_T_beta', action='store_true', dest='mouseTRB', default=False, help='use default mouse TRB model (T cell beta chain)')
    parser.add_option('--humanIGH', '--human_B_heavy', action='store_true', dest='humanIGH', default=False, help='use default human IGH model (B cell heavy chain)')
    parser.add_option('--humanIGK', '--human_B_kappa', action='store_true', dest='humanIGK', default=False, help='use default human IGK model (B cell light kappa chain)')
    parser.add_option('--humanIGL', '--human_B_lambda', action='store_true', dest='humanIGL', default=False, help='use default human IGL model (B cell light lambda chain)')
    parser.add_option('--mouseTRA', '--mouse_T_alpha', action='store_true', dest='mouseTRA', default=False, help='use default mouse TRA model (T cell alpha chain)')
    parser.add_option('--set_custom_model_VDJ', dest='vdj_model_folder', metavar='PATH/TO/FOLDER/', help='specify PATH/TO/FOLDER/ for a custom VDJ generative model')
    parser.add_option('--set_custom_model_VJ', dest='vj_model_folder', metavar='PATH/TO/FOLDER/', help='specify PATH/TO/FOLDER/ for a custom VJ generative model')
    parser.add_option('--sonia_model', type='string', default = 'leftright', dest='model_type' ,help=' specify model type: leftright or lengthpos, default is leftright')
    parser.add_option('--post', '--ppost', action='store_true', dest='ppost', default=False, help='sample from post selected repertoire')
    parser.add_option('--pre', '--pgen', action='store_true', dest='pgen', default=False, help='sample from pre selected repertoire ')
    parser.add_option('--delimiter_out','-d', type='choice', dest='delimiter_out',  choices=['tab', 'space', ',', ';', ':'], help="declare outfile delimiter. Default is tab for .tsv output files, comma for .csv files, and the infile delimiter for all others. Choices: 'tab', 'space', ',', ';', ':'")
    parser.add_option('-s','--chunk_size', type='int',metavar='N', dest='chunck_size', default = int(1e3), help='Number of sequences to generate at each iteration')
    parser.add_option('-r','--rejection_bound', type='int',metavar='N', dest='rejection_bound', default = 10, help='limit above which sequences are always accepted.')

    # input output
    parser.add_option('-o', '--outfile', dest = 'outfile_name', metavar='PATH/TO/FILE', help='write CDR3 sequences to PATH/TO/FILE')
    parser.add_option('-n', '--N', type='int',metavar='N', dest='num_seqs_to_generate',default=1, help='Number of sequences to sample from.')

    (options, args) = parser.parse_args()

    #Check that the model is specified properly
    main_folder = os.path.dirname(__file__)

    default_models = {}
    default_models['humanTRA'] = [os.path.join(main_folder, 'default_models', 'human_T_alpha'),  'VJ']
    default_models['humanTRB'] = [os.path.join(main_folder, 'default_models', 'human_T_beta'), 'VDJ']
    default_models['mouseTRB'] = [os.path.join(main_folder, 'default_models', 'mouse_T_beta'), 'VDJ']
    default_models['humanIGH'] = [os.path.join(main_folder, 'default_models', 'human_B_heavy'), 'VDJ']
    default_models['humanIGK'] = [os.path.join(main_folder, 'default_models', 'human_B_kappa'), 'VJ']
    default_models['humanIGL'] = [os.path.join(main_folder, 'default_models', 'human_B_lambda'),  'VJ']
    default_models['mouseTRA'] = [os.path.join(main_folder, 'default_models', 'mouse_T_alpha'), 'VJ']

    num_models_specified = sum([1 for x in list(default_models.keys()) + ['vj_model_folder', 'vdj_model_folder'] if getattr(options, x)])

    if num_models_specified == 1: #exactly one model specified
        try:
            d_model = [x for x in default_models.keys() if getattr(options, x)][0]
            model_folder = default_models[d_model][0]
            recomb_type = default_models[d_model][1]
        except IndexError:
            if options.vdj_model_folder: #custom VDJ model specified
                model_folder = options.vdj_model_folder
                recomb_type = 'VDJ'
            elif options.vj_model_folder: #custom VJ model specified
                model_folder = options.vj_model_folder
                recomb_type = 'VJ'
    elif num_models_specified == 0:
        print('Need to indicate generative model.')
        print('Exiting...')
        return -1
    elif num_models_specified > 1:
        print('Only specify one model')
        print('Exiting...')
        return -1
    
    #Parse delimiter_out
    delimiter_out = options.delimiter_out
    if delimiter_out is None: #Default case
        delimiter_out = '\t'    
        if options.outfile_name is None:
            pass
        elif options.outfile_name.endswith('.tsv'): #output TAB separated value file
            delimiter_out = '\t'
        elif options.outfile_name.endswith('.csv'): #output COMMA separated value file
            delimiter_out = ','
    else:
        try:
            delimiter_out = {'tab': '\t', 'space': ' ', ',': ',', ';': ';', ':': ':'}[delimiter_out]
        except KeyError:
            pass #Other string passed as the delimiter.
    #Generative model specification -- note we'll probably change this syntax to
    #allow for arbitrary model file specification
    params_file_name = os.path.join(model_folder,'model_params.txt')
    marginals_file_name = os.path.join(model_folder,'model_marginals.txt')
    V_anchor_pos_file = os.path.join(model_folder,'V_gene_CDR3_anchors.csv')
    J_anchor_pos_file = os.path.join(model_folder,'J_gene_CDR3_anchors.csv')

    for x in [params_file_name, marginals_file_name, V_anchor_pos_file, J_anchor_pos_file]:
            if not os.path.isfile(x):
                print('Cannot find: ' + x)
                print('Please check the files (and naming conventions) in the model folder ' + model_folder)
                print('Exiting...')
                return -1

    #Load up model based on recomb_type
    #VDJ recomb case --- used for TCRB and IGH
    if recomb_type == 'VDJ':
        genomic_data = olga_load_model.GenomicDataVDJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        generative_model = olga_load_model.GenerativeModelVDJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        seqgen_model = sequence_generation.SequenceGenerationVDJ(generative_model, genomic_data)
    #VJ recomb case --- used for TCRA and light chain
    elif recomb_type == 'VJ':
        genomic_data = olga_load_model.GenomicDataVJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        generative_model = olga_load_model.GenerativeModelVJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        seqgen_model = sequence_generation.SequenceGenerationVJ(generative_model, genomic_data)

    if options.pgen:sonia_model=SoniaLeftposRightpos()
    else:sonia_model=SoniaLeftposRightpos(feature_file=os.path.join(model_folder,'features.tsv'),log_file=os.path.join(model_folder,'log.txt'),vj=recomb_type == 'VJ')
    
    # load Evaluate model class
    seq_gen=SequenceGeneration(sonia_model,custom_olga_model=seqgen_model,custom_genomic_data=genomic_data)

    if options.outfile_name is not None: #OUTFILE SPECIFIED
        with open(options.outfile_name,'w') as file:
            to_generate=chuncks(options.num_seqs_to_generate,options.chunck_size)
            for t in tqdm(to_generate):
                if options.pgen:
                    seqs=seq_gen.generate_sequences_pre(num_seqs=t,nucleotide=True)
                elif options.ppost:
                    seqs=seq_gen.generate_sequences_post(num_seqs=t,nucleotide=True,upper_bound=options.rejection_bound)
                else: 
                    print ('ERROR: give option between --pre or --post')
                    return -1
                for seq in seqs: file.write(seq[0]+delimiter_out+seq[1]+delimiter_out+seq[2]+delimiter_out+seq[3]+'\n')
       # np.savetxt(options.outfile_name,seqs,fmt='%s')

    else: #print to stdout
        to_generate=chuncks(options.num_seqs_to_generate,options.chunck_size)
        for t in to_generate:
            if options.pgen:
                seqs=seq_gen.generate_sequences_pre(num_seqs=t,nucleotide=True)
            elif options.ppost:
                seqs=seq_gen.generate_sequences_post(num_seqs=t,nucleotide=True,upper_bound=options.rejection_bound)
            else:
                print ('ERROR: give option between --pre or --post')
                return -1
            for seq in seqs:
                print(seq[0],seq[1],seq[2],seq[3])

if __name__ == '__main__': main()
