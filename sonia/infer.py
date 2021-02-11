#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command line script to infer sonia model.

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

This program will infer a seleciton model
"""

from __future__ import print_function, division,absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from optparse import OptionParser
import olga.sequence_generation as sequence_generation
from sonia.sonia_length_pos import SoniaLengthPos
from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
from sonia.evaluate_model import EvaluateModel
import time
from sonia.utils import gene_to_num_str
import olga.load_model as olga_load_model
import olga.generation_probability as generation_probability
import numpy as np
from tqdm import tqdm


#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass

def main():
    """ Evaluate sequences."""
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
    parser.add_option('--sonia_model', type='string', default = 'leftright', dest='model_type' ,help='specify model type: leftright or lengthpos, default is leftright')
    parser.add_option('--epochs', type='int', default = 30, dest='epochs' ,help='number of epochs for inference, default is 30')
    parser.add_option('--batch_size', type='int', default = 5000, dest='batch_size' ,help='size of batch for the stochastic gradient descent')
    parser.add_option('--validation_split', type='float', default = 0.2, dest='validation_split' ,help='fraction of sequences used for validation.')
    parser.add_option('--independent_genes', '--include_indep_genes', action='store_true', dest='independent_genes', default=False, help='Independent gene selection factors q_v*q_j. Deafult is joint q_vj')
    parser.add_option('--min_energy_clip', type='float', default=-5, dest='min_energy_clip',  help='Set numerical lower bound to the values of -logQ, default is -5.')
    parser.add_option('--max_energy_clip', type='float', default=10, dest='max_energy_clip', help='Set numerical upper bound to the values of -logQ, default is 10.')

    #location of seqs
    parser.add_option('--seq_in', '--seq_index', type='int', metavar='INDEX', dest='seq_in_index', default = 0, help='specifies sequences to be read in are in column INDEX. Default is index 0 (the first column).')
    parser.add_option('--v_in', '--v_mask_index', type='int', metavar='INDEX', dest='V_mask_index', default=1, help='specifies V_masks are found in column INDEX in the input file. Default is 1.')
    parser.add_option('--j_in', '--j_mask_index', type='int', metavar='INDEX', dest='J_mask_index', default=2, help='specifies J_masks are found in column INDEX in the input file. Default is 2.')

    # input output
    parser.add_option('-i', '--infile', dest = 'infile_name',metavar='PATH/TO/FILE', help='read in CDR3 sequences (and optionally V/J masks) from PATH/TO/FILE')
    parser.add_option('-o', '--outfile', dest = 'outfile_name', metavar='PATH/TO/FILE', help='write CDR3 sequences and pgens to PATH/TO/FILE')
    parser.add_option('-m', '--max_number_of_seqs', type='int',metavar='N', dest='max_number_of_seqs', help='evaluate for at most N sequences.')
    parser.add_option('-n', '--n_gen_seqs', type='int',metavar='N', dest='n_gen_seqs',default=0, help='sample n sequences from gen distribution.')
    parser.add_option('-g', '--infile_gen', dest = 'infile_gen',metavar='PATH/TO/FILE', help='read generated CDR3 sequences (and optionally V/J masks) from PATH/TO/FILE')
    parser.add_option('--lines_to_skip', type='int',metavar='N', dest='lines_to_skip', default = 0, help='skip the first N lines of the file. Default is 0.')
    parser.add_option('--no_report', '--no_plot_report', action='store_false', dest='plot_report', default=True, help='Do not produce report plots of the inferred model.')
    
    #delimeters
    parser.add_option('-d', '--delimiter', type='choice', dest='delimiter',  choices=['tab', 'space', ',', ';', ':'], help="declare infile delimiter. Default is tab for .tsv input files, comma for .csv files, and any whitespace for all others. Choices: 'tab', 'space', ',', ';', ':'")
    parser.add_option('--raw_delimiter', type='str', dest='delimiter', help="declare infile delimiter as a raw string.")
    parser.add_option('--delimiter_out', type='choice', dest='delimiter_out',  choices=['tab', 'space', ',', ';', ':'], help="declare outfile delimiter. Default is tab for .tsv output files, comma for .csv files, and the infile delimiter for all others. Choices: 'tab', 'space', ',', ';', ':'")
    parser.add_option('--raw_delimiter_out', type='str', dest='delimiter_out', help="declare for the delimiter outfile as a raw string.")
    parser.add_option('--gene_mask_delimiter', type='choice', dest='gene_mask_delimiter',  choices=['tab', 'space', ',', ';', ':'], help="declare gene mask delimiter. Default comma unless infile delimiter is comma, then default is a semicolon. Choices: 'tab', 'space', ',', ';', ':'")
    parser.add_option('--raw_gene_mask_delimiter', type='str', dest='gene_mask_delimiter', help="declare delimiter of gene masks as a raw string.")
    parser.add_option('--comment_delimiter', type='str', dest='comment_delimiter', help="character or string to indicate comment or header lines to skip.")
    parser.add_option('--seed', type='int',metavar='N', dest='seed', default = None, help='set seed for inference')

    (options, args) = parser.parse_args()

    #set seed
    if options.seed is not None: 
        import tensorflow as tf
        np.random.seed(options.seed)
        tf.random.set_seed(options.seed)

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

    if options.independent_genes:
        independent_genes=True
        joint_genes=False
    else:
        independent_genes=False
        joint_genes=True

    num_models_specified = sum([1 for x in list(default_models.keys()) + ['vj_model_folder', 'vdj_model_folder'] if getattr(options, x)])
    recompute_productive_norm=False
    if num_models_specified == 1: #exactly one model specified
        try:
            d_model = [x for x in default_models.keys() if getattr(options, x)][0]
            model_folder = default_models[d_model][0]
            recomb_type = default_models[d_model][1]
        except IndexError:
            if options.vdj_model_folder: #custom VDJ model specified
                recompute_productive_norm=True
                model_folder = options.vdj_model_folder
                recomb_type = 'VDJ'
            elif options.vj_model_folder: #custom VJ model specified
                recompute_productive_norm=True
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
    
    if options.max_energy_clip <= options.min_energy_clip :
        print('The clip for the higher energy must be strictly greater than the clip for the lower energy. ')
        print('Exiting...')
        return -1
    else :
        max_energy_clip = options.max_energy_clip
        min_energy_clip = options.min_energy_clip

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
        pgen_model = generation_probability.GenerationProbabilityVDJ(generative_model, genomic_data)
    #VJ recomb case --- used for TCRA and light chain
    elif recomb_type == 'VJ':
        genomic_data = olga_load_model.GenomicDataVJ()
        genomic_data.load_igor_genomic_data(params_file_name, V_anchor_pos_file, J_anchor_pos_file)
        generative_model = olga_load_model.GenerativeModelVJ()
        generative_model.load_and_process_igor_model(marginals_file_name)
        pgen_model = generation_probability.GenerationProbabilityVJ(generative_model, genomic_data)

    if options.infile_name is not None:
        infile_name = options.infile_name

        if not os.path.isfile(infile_name):
            print('Cannot find input file: ' + infile_name)
            print('Exiting...')
            return -1

    if options.outfile_name is not None:
        outfile_name = options.outfile_name
        if os.path.isfile(outfile_name):
            if not input(outfile_name + ' already exists. Overwrite (y/n)? ').strip().lower() in ['y', 'yes']:
                print('Exiting...')
                return -1

    #Parse delimiter
    delimiter = options.delimiter
    if delimiter is None: #Default case
        if options.infile_name is None:
            delimiter = '\t'
        elif infile_name.endswith('.tsv'): #parse TAB separated value file
            delimiter = '\t'
        elif infile_name.endswith('.csv'): #parse COMMA separated value file
            delimiter = ','
    else:
        try:
            delimiter = {'tab': '\t', 'space': ' ', ',': ',', ';': ';', ':': ':'}[delimiter]
        except KeyError:
            pass #Other string passed as the delimiter.

    #Parse delimiter_out
    delimiter_out = options.delimiter_out
    if delimiter_out is None: #Default case
        if delimiter is None:
            delimiter_out = '\t'
        else:
            delimiter_out = delimiter
        if options.outfile_name is None:
            pass
        elif outfile_name.endswith('.tsv'): #output TAB separated value file
            delimiter_out = '\t'
        elif outfile_name.endswith('.csv'): #output COMMA separated value file
            delimiter_out = ','
    else:
        try:
            delimiter_out = {'tab': '\t', 'space': ' ', ',': ',', ';': ';', ':': ':'}[delimiter_out]
        except KeyError:
            pass #Other string passed as the delimiter.

    #Parse gene_delimiter
    gene_mask_delimiter = options.gene_mask_delimiter
    if gene_mask_delimiter is None: #Default case
        gene_mask_delimiter = ','
        if delimiter == ',':
            gene_mask_delimiter = ';'
    else:
        try:
            gene_mask_delimiter = {'tab': '\t', 'space': ' ', ',': ',', ';': ';', ':': ':'}[gene_mask_delimiter]
        except KeyError:
            pass #Other string passed as the delimiter.

    #More options
    seq_in_index = options.seq_in_index #where in the line the sequence is after line.split(delimiter)
    lines_to_skip = options.lines_to_skip #one method of skipping header
    comment_delimiter = options.comment_delimiter #another method of skipping header
    max_number_of_seqs = options.max_number_of_seqs
    V_mask_index = options.V_mask_index #Default is not conditioning on V identity
    J_mask_index = options.J_mask_index #Default is not conditioning on J identity
    skip_empty=True # skip empty lines
    if options.infile_name is None: #No infile specified -- args should be the input seqs
        print('ERROR: specify input file.')
        return -1
    else:
        seqs = []
        V_usage_masks = []
        J_usage_masks = []
        print('Read input file.')
        infile = open(infile_name, 'r')

        for i, line in enumerate(tqdm(infile)):
            if comment_delimiter is not None: #Default case -- no comments/header delimiter
                if line.startswith(comment_delimiter): #allow comments
                    continue
            if i < lines_to_skip:
                continue

            if delimiter is None: #Default delimiter is any whitespace
                split_line = line.split('\n')[0].split()
            else:
                split_line = line.split('\n')[0].split(delimiter)
            #Find the seq
            try:
                seq = split_line[seq_in_index].strip()
                if len(seq.strip()) == 0:
                    if skip_empty:
                        continue
                    else:
                        seqs.append(seq) #keep the blank seq as a placeholder
                        #seq_types.append('aaseq')
                else:
                    seqs.append(seq)
                    #seq_types.append(determine_seq_type(seq, aa_alphabet))
            except IndexError: #no index match for seq
                if skip_empty and len(line.strip()) == 0:
                    continue
                print('seq_in_index is out of range')
                print('Exiting...')
                infile.close()
                return -1

            #Find and format V_usage_mask
            if V_mask_index is None:
                V_usage_masks.append(None) #default mask
            else:
                try:
                    V_usage_mask = split_line[V_mask_index].strip().split(gene_mask_delimiter)
                    #check that all V gene/allele names are recognized
                    if all([gene_to_num_str(v, 'V') in pgen_model.V_mask_mapping for v in V_usage_mask]):
                        V_usage_masks.append(V_usage_mask)
                    else:
                        print(str(V_usage_mask) + " is not a usable V_usage_mask composed exclusively of recognized V gene/allele names")
                        print('Unrecognized V gene/allele names: ' + ', '.join([v for v in V_usage_mask if gene_to_num_str(v, 'V') not in pgen_model.V_mask_mapping.keys()]))
                        print('Continuing but inference might be biased...')
                        V_usage_masks.append(V_usage_mask)
                        #infile.close()
                        #return -1
                except IndexError: #no index match for V_mask_index
                    print('V_mask_index is out of range, check the delimeter.')
                    print('Exiting...')
                    infile.close()
                    return -1

            #Find and format J_usage_mask
            if J_mask_index is None:
                J_usage_masks.append(None) #default mask
            else:
                try:
                    J_usage_mask = split_line[J_mask_index].strip().split(gene_mask_delimiter)
                    #check that all V gene/allele names are recognized
                    if all([gene_to_num_str(j, 'J') in pgen_model.J_mask_mapping for j in J_usage_mask]):
                        J_usage_masks.append(J_usage_mask)
                    else:
                        print(str(J_usage_mask) + " is not a usable J_usage_mask composed exclusively of recognized J gene/allele names")
                        print('Unrecognized J gene/allele names: ' + ', '.join([j for j in J_usage_mask if gene_to_num_str(j, 'J') not in pgen_model.J_mask_mapping.keys()]))
                        print('Continuing but inference might be biased...')
                        J_usage_masks.append(J_usage_mask)

                        #infile.close()
                        #return -1
                except IndexError: #no index match for J_mask_index
                    print('J_mask_index is out of range, check the delimeter.')
                    print('Exiting...')
                    infile.close()
                    return -1

            if max_number_of_seqs is not None:
                if len(seqs) >= max_number_of_seqs:
                    break

        data_seqs=[[seqs[i],V_usage_masks[i][0],J_usage_masks[i][0]] for i in range(len(seqs))]
        #define number of gen_seqs:
        gen_seqs=[]
        n_gen_seqs=options.n_gen_seqs
        generate_sequences=False
        if options.infile_gen is None:
            generate_sequences=True
            if n_gen_seqs is 0: n_gen_seqs=np.max([int(3e5),3*len(data_seqs)])
        else:
            seqs = []
            V_usage_masks = []
            J_usage_masks = []
            print('Read file of generated seqs.')
            infile = open(options.infile_gen, 'r')

            for i, line in enumerate(tqdm(infile)):
                if comment_delimiter is not None: #Default case -- no comments/header delimiter
                    if line.startswith(comment_delimiter): #allow comments
                        continue
                if i < lines_to_skip:
                    continue

                if delimiter is None: #Default delimiter is any whitespace
                    split_line = line.split('\n')[0].split()
                else:
                    split_line = line.split('\n')[0].split(delimiter)
                #Find the seq
                try:
                    seq = split_line[seq_in_index].strip()
                    if len(seq.strip()) == 0:
                        if skip_empty:
                            continue
                        else:
                            seqs.append(seq) #keep the blank seq as a placeholder
                            #seq_types.append('aaseq')
                    else:
                        seqs.append(seq)
                        #seq_types.append(determine_seq_type(seq, aa_alphabet))
                except IndexError: #no index match for seq
                    if skip_empty and len(line.strip()) == 0:
                        continue
                    print('seq_in_index is out of range')
                    print('Exiting...')
                    infile.close()
                    return -1

                #Find and format V_usage_mask
                if V_mask_index is None:
                    V_usage_masks.append(None) #default mask
                else:
                    try:
                        V_usage_mask = split_line[V_mask_index].strip().split(gene_mask_delimiter)
                        #check that all V gene/allele names are recognized
                        if all([gene_to_num_str(v, 'V') in pgen_model.V_mask_mapping for v in V_usage_mask]):
                            V_usage_masks.append(V_usage_mask)
                        else:
                            print(str(V_usage_mask) + " is not a usable V_usage_mask composed exclusively of recognized V gene/allele names")
                            print('Unrecognized V gene/allele names: ' + ', '.join([v for v in V_usage_mask if gene_to_num_str(v, 'V') not in pgen_model.V_mask_mapping.keys()]))
                            print('Continuing but inference might be biased...')
                            V_usage_masks.append(V_usage_mask)
                            #infile.close()
                            #return -1
                    except IndexError: #no index match for V_mask_index
                        print('V_mask_index is out of range, check the delimeter.')
                        print('Exiting...')
                        infile.close()
                        return -1

                #Find and format J_usage_mask
                if J_mask_index is None:
                    J_usage_masks.append(None) #default mask
                else:
                    try:
                        J_usage_mask = split_line[J_mask_index].strip().split(gene_mask_delimiter)
                        #check that all V gene/allele names are recognized
                        if all([gene_to_num_str(j, 'J') in pgen_model.J_mask_mapping for j in J_usage_mask]):
                            J_usage_masks.append(J_usage_mask)
                        else:
                            print(str(J_usage_mask) + " is not a usable J_usage_mask composed exclusively of recognized J gene/allele names")
                            print('Unrecognized J gene/allele names: ' + ', '.join([j for j in J_usage_mask if gene_to_num_str(j, 'J') not in pgen_model.J_mask_mapping.keys()]))
                            print('Continuing but inference might be biased...')
                            J_usage_masks.append(J_usage_mask)

                            #infile.close()
                            #return -1
                    except IndexError: #no index match for J_mask_index
                        print('J_mask_index is out of range, check the delimeter.')
                        print('Exiting...')
                        infile.close()
                        return -1

            gen_seqs=[[seqs[i],V_usage_masks[i][0],J_usage_masks[i][0]] for i in range(len(seqs))]
        # combine sequences.
        print('Initialise Model.')

        # choose sonia model type
        if options.model_type=='leftright': 
            sonia_model=SoniaLeftposRightpos(data_seqs=data_seqs,
                                             gen_seqs=gen_seqs,
                                             custom_pgen_model=model_folder,
                                             vj=recomb_type == 'VJ',
                                             include_joint_genes=joint_genes,
                                             include_indep_genes=independent_genes,
                                             min_energy_clip=min_energy_clip,
                                             max_energy_clip=max_energy_clip
                                            )
        elif options.model_type=='lengthpos':
            sonia_model=SoniaLengthPos(data_seqs=data_seqs,
                                       gen_seqs=gen_seqs,
                                       custom_pgen_model=model_folder,
                                       vj=recomb_type == 'VJ',
                                       include_joint_genes=joint_genes,
                                       include_indep_genes=independent_genes,
                                       min_energy_clip=min_energy_clip,
                                       max_energy_clip=max_energy_clip
                                      )
        else:
            print('ERROR: choose a model between leftright or lengthpos')

        if generate_sequences: sonia_model.add_generated_seqs(n_gen_seqs,custom_model_folder=model_folder) 

        if recompute_productive_norm: sonia_model.norm_productive=pgen_model.compute_regex_CDR3_template_pgen('CX{0,}')
        
        print('Model initialised. Start inference')
        sonia_model.infer_selection(epochs=options.epochs,verbose=1,batch_size=options.batch_size,validation_split=options.validation_split)
        print('Save Model')
        if options.outfile_name is not None: #OUTFILE SPECIFIED
            sonia_model.save_model(options.outfile_name)
            if options.plot_report:
                from sonia.plotting import Plotter
                pl=Plotter(sonia_model)
                pl.plot_model_learning(os.path.join(options.outfile_name, 'model_learning.png'))
                pl.plot_vjl(os.path.join(options.outfile_name, 'marginals.png'))
                pl.plot_logQ(os.path.join(options.outfile_name, 'log_Q.png'))
                pl.plot_ratioQ(os.path.join(options.outfile_name, 'Q_ratio.png'))

        else: #print to stdout
            sonia_model.save_model('sonia_model')
            if options.plot_report:
                from sonia.plotting import Plotter
                pl=Plotter(sonia_model)
                pl.plot_model_learning(os.path.join('sonia_model', 'model_learning.png'))
                pl.plot_vjl(os.path.join('sonia_model', 'marginals.png'))
                pl.plot_logQ(os.path.join('sonia_model', 'log_Q.png'))
                pl.plot_ratioQ(os.path.join('sonia_model', 'Q_ratio.png'))

if __name__ == '__main__': main()
