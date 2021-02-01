#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command line script to evaluate sequences.

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

This program will evaluate pgen and ppost of sequences
"""

from __future__ import print_function, division,absolute_import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from optparse import OptionParser
from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
from sonia.evaluate_model import EvaluateModel
from sonia.utils import gene_to_num_str
import olga.load_model as olga_load_model
import olga.generation_probability as generation_probability
import numpy as np
import time
from tqdm import tqdm
import multiprocessing as mp

#Set input = raw_input for python 2
try:
    import __builtin__
    input = getattr(__builtin__, 'raw_input')
except (ImportError, AttributeError):
    pass

def chunks(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

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
    parser.add_option('--sonia_model', type='string', default = 'leftright', dest='model_type' ,help=' specify model type: leftright or lengthpos, default is leftright')
    parser.add_option('--ppost', '--Ppost', action='store_true', dest='ppost', default=False, help='compute Ppost, also computes pgen and Q')
    parser.add_option('--pgen', '--Pgen', action='store_true', dest='pgen', default=False, help='compute pgen')
    parser.add_option('--Q', '--selection_factor', action='store_true', dest='Q', default=False, help='compute Q')
    parser.add_option('--recompute_productive_norm', '--compute_norm', action='store_true', dest='recompute_productive_norm', default=False, help='recompute productive normalization')
    parser.add_option('--skip_off','--skip_empty_off', action='store_true', dest = 'skip_empty', default=True, help='stop skipping empty or blank sequences/lines (if for example you want to keep line index fidelity between the infile and outfile).')

    parser.add_option('-s','--chunk_size', type='int',metavar='N', dest='chunck_size', default = mp.cpu_count()*int(5e2), help='Number of sequences to evaluate at each iteration')

    #vj genes
    parser.add_option('--v_in', '--v_mask_index', type='int', metavar='INDEX', dest='V_mask_index', default=1, help='specifies V_masks are found in column INDEX in the input file. Default is 1.')
    parser.add_option('--j_in', '--j_mask_index', type='int', metavar='INDEX', dest='J_mask_index', default=2, help='specifies J_masks are found in column INDEX in the input file. Default is 2.')
    parser.add_option('--v_mask', type='string', dest='V_mask', help='specify V usage to condition as arguments.')
    parser.add_option('--j_mask', type='string', dest='J_mask', help='specify J usage to condition as arguments.')

    # input output
    parser.add_option('-i', '--infile', dest = 'infile_name',metavar='PATH/TO/FILE', help='read in CDR3 sequences (and optionally V/J masks) from PATH/TO/FILE')
    parser.add_option('-o', '--outfile', dest = 'outfile_name', metavar='PATH/TO/FILE', help='write CDR3 sequences and pgens to PATH/TO/FILE')
    parser.add_option('--seq_in', '--seq_index', type='int', metavar='INDEX', dest='seq_in_index', default = 0, help='specifies sequences to be read in are in column INDEX. Default is index 0 (the first column).')
    parser.add_option('-m', '--max_number_of_seqs', type='int',metavar='N', dest='max_number_of_seqs', help='evaluate for at most N sequences.')
    parser.add_option('--lines_to_skip', type='int',metavar='N', dest='lines_to_skip', default = 0, help='skip the first N lines of the file. Default is 0.')

    
    #delimeters
    parser.add_option('-d', '--delimiter', type='choice', dest='delimiter',  choices=['tab', 'space', ',', ';', ':'], help="declare infile delimiter. Default is tab for .tsv input files, comma for .csv files, and any whitespace for all others. Choices: 'tab', 'space', ',', ';', ':'")
    parser.add_option('--raw_delimiter', type='str', dest='delimiter', help="declare infile delimiter as a raw string.")
    parser.add_option('--delimiter_out', type='choice', dest='delimiter_out',  choices=['tab', 'space', ',', ';', ':'], help="declare outfile delimiter. Default is tab for .tsv output files, comma for .csv files, and the infile delimiter for all others. Choices: 'tab', 'space', ',', ';', ':'")
    parser.add_option('--raw_delimiter_out', type='str', dest='delimiter_out', help="declare for the delimiter outfile as a raw string.")
    parser.add_option('--gene_mask_delimiter', type='choice', dest='gene_mask_delimiter',  choices=['tab', 'space', ',', ';', ':'], help="declare gene mask delimiter. Default comma unless infile delimiter is comma, then default is a semicolon. Choices: 'tab', 'space', ',', ';', ':'")
    parser.add_option('--raw_gene_mask_delimiter', type='str', dest='gene_mask_delimiter', help="declare delimiter of gene masks as a raw string.")
    parser.add_option('--comment_delimiter', type='str', dest='comment_delimiter', help="character or string to indicate comment or header lines to skip.")

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
#        if os.path.isfile(outfile_name):
#            if not input(outfile_name + ' already exists. Overwrite (y/n)? ').strip().lower() in ['y', 'yes']:
#                print('Exiting...')
#                return -1

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
    skip_empty = options.skip_empty

    #print(V_mask_index,J_mask_index,seq_in_index,gene_mask_delimiter,delimiter)
    
    # choose sonia model type
    sonia_model=SoniaLeftposRightpos(feature_file=os.path.join(model_folder,'features.tsv'),log_file=os.path.join(model_folder,'log.txt'),vj=recomb_type == 'VJ',custom_pgen_model=model_folder)
    if options.recompute_productive_norm: 
        print('Recompute productive normalization.')
        sonia_model.norm_productive=pgen_model.compute_regex_CDR3_template_pgen('CX{0,}')

    # load Evaluate model class
    ev=EvaluateModel(sonia_model,custom_olga_model=pgen_model)

    if options.infile_name is None: #No infile specified -- args should be the input seq
        print_warnings = True
        if len(args)>1 : 
            print('ERROR: can process only one sequence at the time. Submit thourgh file instead.')
            return -1
        seq=args[0]

        #Format V and J masks -- uniform for all argument input sequences
 
        try:
            V_mask = options.V_mask.split(',')
            unrecognized_v_genes = [v for v in V_mask if gene_to_num_str(v, 'V') not in pgen_model.V_mask_mapping.keys()]
            V_mask = [v for v in V_mask if gene_to_num_str(v, 'V') in pgen_model.V_mask_mapping.keys()]
            if len(unrecognized_v_genes) > 0:
                print('These V genes/alleles are not recognized: ' + ', '.join(unrecognized_v_genes))
            if len(V_mask) == 0:
                print('No recognized V genes/alleles in the provided V_mask. Continuing without conditioning on V usage.')
                V_mask = None
        except AttributeError:
            V_mask = options.V_mask #Default is None, i.e. not conditioning on V identity

        try:
            J_mask = options.J_mask.split(',')
            unrecognized_j_genes = [j for j in J_mask if gene_to_num_str(j, 'J') not in pgen_model.J_mask_mapping.keys()]
            J_mask = [j for j in J_mask if gene_to_num_str(j, 'J') in pgen_model.J_mask_mapping.keys()]
            if len(unrecognized_j_genes) > 0:
                print('These J genes/alleles are not recognized: ' + ', '.join(unrecognized_j_genes))
            if len(J_mask) == 0:
                print('No recognized J genes/alleles in the provided J_mask. Continuing without conditioning on J usage.')
                J_mask = None
        except AttributeError:
            J_mask = options.J_mask #Default is None, i.e. not conditioning on J identity

        print('')

        if options.ppost:
            if options.V_mask is None: V_mask=['']
            if options.J_mask is None: J_mask=['']

            v,j=V_mask[0],J_mask[0]
            Q,pgen,ppost=ev.evaluate_seqs([[seq,v,j]])
            print('Ppost of ' + seq + ' '+v+ ' '+j+ ': ' + str(ppost[0]))
            print('Pgen of ' + seq + ' '+v+ ' '+j+ ': ' + str(pgen[0]))
            print('Q of ' + seq + ' '+v+ ' '+j+ ': ' + str(Q[0]))
            print('')
        elif options.Q:
            if options.V_mask is None: V_mask=['']
            if options.J_mask is None: J_mask=['']
            v,j=V_mask[0],J_mask[0]
            Q=ev.evaluate_selection_factors([[seq,v,j]])
            print('Q of ' + seq + ' '+v+ ' '+j+ ': ' + str(Q[0]))
        elif options.pgen:
            pgen=pgen_model.compute_aa_CDR3_pgen(seq,V_mask,J_mask)
            if J_mask is None: J_mask= ''
            if V_mask is None: V_mask= ''
            print('Pgen of ' + seq + ' '+','.join(V_mask)+ ' '+','.join(J_mask)+ ': ' + str(pgen))

        else:
            print('Specify and option: --ppost, --pgen or --Q')


    else:
        print('Load file')

        seqs = []
        V_usage_masks = []
        J_usage_masks = []

        infile = open(infile_name, 'r')

        for i, line in enumerate(infile):
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
                        print('Exiting...')
                        infile.close()
                        return -1
                except IndexError: #no index match for V_mask_index
                    print('V_mask_index is out of range')
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
                        print('Exiting...')
                        infile.close()
                        return -1
                except IndexError: #no index match for J_mask_index
                    print('J_mask_index is out of range')
                    print('Exiting...')
                    infile.close()
                    return -1

            if max_number_of_seqs is not None:
                if len(seqs) >= max_number_of_seqs:
                    break

        # combine sequences.
        zipped=[[seqs[i],V_usage_masks[i][0],J_usage_masks[i][0]] for i in range(len(seqs))]

        print('Evaluate')

        if options.outfile_name is not None: #OUTFILE SPECIFIED
            with open(options.outfile_name,'w') as file:
                if options.ppost:file.write('Q'+delimiter_out+'Pgen'+delimiter_out+'Ppost\n')
                elif options.Q:file.write('Q\n')
                elif options.pgen:file.write('Pgen\n')
                else:
                    print('Specify one option: --ppost, --pgen or --Q')
                    return -1
                for t in tqdm(chunks(zipped,options.chunck_size)):
                    if options.ppost:
                        Q,pgen,ppost=ev.evaluate_seqs(t)
                        for i in range(len(Q)):file.write(str(Q[i])+delimiter_out+str(pgen[i])+delimiter_out+str(ppost[i])+'\n')
                    elif options.Q:
                        Q=ev.evaluate_selection_factors(t)
                        for i in range(len(Q)):file.write(str(Q[i])+'\n')
                    elif options.pgen:
                        pgens=ev.compute_all_pgens(t)
                        for i in range(len(pgens)):file.write(str(pgens[i])+'\n')

        else: #print to stdout
            for t in chunks(zipped,options.chunck_size):
                if options.ppost:
                    Q,pgen,ppost=ev.evaluate_seqs(t)
                    print ('Q, Pgen, Ppost')
                    for i in range(len(Q)):print(Q[i],pgen[i],ppost[i])
                elif options.Q:
                    Q=ev.evaluate_selection_factors(t)
                    print ('Q')
                    print(Q)
                elif options.pgen:
                    pgens=ev.compute_all_pgens(t)
                    print ('Pgen')
                    print(pgens)
                else:
                    print('Specify one option: --ppost, --pgen or --Q')


if __name__ == '__main__': main()