from __future__ import division

import os
import pickle
import sys

import _config  # , _data
import _lib

sys.path.append('/cluster/mshen/')
from collections import defaultdict
from mylib import util
import pandas as pd
import matplotlib
import re

matplotlib.use('Pdf')

# Default params
DEFAULT_INP_DIR = None
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
redo = False

##
# Going wide: experiments to analyze
##
exps = ['Lib1-mES-controladj',
        'Lib1-HEK293T-controladj',
        'Lib1-HCT116-controladj',
        'DisLib-mES-controladj',
        'DisLib-HEK293T',
        'DisLib-U2OS-controladj',
        '0226-PRLmESC-Lib1-Cas9',
        '0226-PRLmESC-Dislib-Cas9',
        'VO-spacers-HEK293-48h-controladj',
        'VO-spacers-HCT116-48h-controladj',
        'VO-spacers-K562-48h-controladj',
        ]


##
# Run statistics
##
def calc_statistics(df, exp, alldf_dict, seq):
    # Calculate statistics on df, saving to alldf_dict
    # Deletion positions

    # Denominator is crispr activity
    df = _lib.crispr_subset(df)
    total_count = sum(df['countEvents'])
    if total_count <= 500:
        return
    df['Frequency'] = _lib.normalize_frequency(df)

    criteria = (df['Type'] == 'INSERTION') & (df['Length'] == 1)
    if sum(df[criteria]['countEvents']) <= 100:
        return
    freq = sum(df[criteria]['Frequency'])
    alldf_dict['Frequency'].append(freq)

    df['insertion_sequence'] = [seq[1].split("+")[1] for seq in df['key_0']]

    s = df[criteria]

    try:
        a_frac = sum(s[s['insertion_sequence'] == 'A']['Frequency']) / freq
    except TypeError:
        a_frac = 0
    alldf_dict['A frac'].append(a_frac)

    try:
        c_frac = sum(s[s['insertion_sequence'] == 'C']['Frequency']) / freq
    except:
        c_frac = 0
    alldf_dict['C frac'].append(c_frac)

    try:
        g_frac = sum(s[s['insertion_sequence'] == 'G']['Frequency']) / freq
    except:
        g_frac = 0
    alldf_dict['G frac'].append(g_frac)

    try:
        t_frac = sum(s[s['insertion_sequence'] == 'T']['Frequency']) / freq
    except:
        t_frac = 0
    alldf_dict['T frac'].append(t_frac)
    # seq IS REPLACED BY NEW INPUT!
    subseq, cutsite = _lib.get_sequence_cutsite(df)
    fivebase = seq[cutsite - 1]
    alldf_dict['Base'].append(fivebase)

    alldf_dict['_Experiment'].append(exp)

    return alldf_dict


def prepare_statistics(data_nm):
    # Input: Dataset
    # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
    # Calculate statistics associated with each experiment by name

    alldf_dict = defaultdict(list)

    dataset = pickle.load(open("../pickle_data/inDelphi_counts_and_deletion_features.pkl", "rb"))
    if dataset is None:
        return

    timer = util.Timer(total=len(dataset))
    # for exp in dataset.keys()[:100]:
    dataset = pd.merge(dataset['counts'], dataset['del_features'],
                       left_on=dataset['counts'].index, right_on=dataset['del_features'].index,
                       how="left")
    dataset['Length'] = [int(x[1].split("+")[1]) if x[1].split("+")[1].isdigit() else len(x[1].split("+")[1]) for x in
                         dataset["key_0"]]
    dataset['cutSite'] = [int(x[1].split("+")[0]) + 29 for x in dataset["key_0"]]
    dataset['exp'] = [x[0] for x in dataset["key_0"]]
    exps = list(set(dataset['exp']))

    # TODO ignores cutsites not compatible with liba, is this correct?
    dataset = dataset[dataset["cutSite"] > 4]
    with open("../data_libprocessing/targets-libA.txt") as f:
        full_dna_exps = []
        for line in f:
            full_dna_exps.append(line.strip("\n"))
    for i, exp in enumerate(exps):
        if i % 500 == 0:
            print(f"Prepare statistics for experiment {i}/{len(exps)}...")
        df = dataset[dataset["exp"] == exp]
        exp_substring_dna = exp[-20:]
        matched_dna = list(filter(lambda x: exp_substring_dna in x, full_dna_exps))
        if matched_dna:
            seq = matched_dna[0]
            calc_statistics(df, exp, alldf_dict, seq)
        else:
            print(f"Experiment {exp} not in libA!")
        timer.update()

    # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
    alldf = pd.DataFrame(alldf_dict)
    return alldf


##
# Load statistics from csv, or calculate 
##
def load_statistics(data_nm):
    print(data_nm)
    out_dir = "./cluster/mshen/prj/mmej_figures/out/"
    stats_csv_fn = out_dir + '1bpins_stats.csv'
    if not os.path.isfile(stats_csv_fn) or redo:
        print('Running statistics from scratch...')
        stats_csv = prepare_statistics(data_nm)
        stats_csv.to_csv(stats_csv_fn)
    else:
        # print('Getting statistics from file...')
        stats_csv = pd.read_csv(stats_csv_fn, index_col=0)
    # print('Done')
    return stats_csv


##
# Plotters
##
def plot():
    # Frequency of deletions by length and MH basis.

    return


##
# qsubs
##
def gen_qsubs():
    # Generate qsub shell scripts and commands for easy parallelization
    print('Generating qsub scripts...')
    qsubs_dir = _config.QSUBS_DIR + NAME + '/'
    util.ensure_dir_exists(qsubs_dir)
    qsub_commands = []

    num_scripts = 0
    for exp in exps:
        command = 'python %s.py %s redo' % (NAME, exp)
        script_id = NAME.split('_')[0]

        # Write shell scripts
        sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, exp)
        with open(sh_fn, 'w') as f:
            f.write('#!/bin/bash\n%s\n' % (command))
        num_scripts += 1

        # Write qsub commands
        qsub_commands.append('qsub -m e -V -wd %s %s' % (_config.SRC_DIR, sh_fn))

    # Save commands
    with open(qsubs_dir + '_commands.txt', 'w') as f:
        f.write('\n'.join(qsub_commands))

    print('Wrote %s shell scripts to %s', (num_scripts, qsubs_dir))
    return


##
# Main
##
@util.time_dec
def main(data_nm='', redo_flag=''):
    print(NAME)
    global out_dir
    util.ensure_dir_exists(out_dir)

    if redo_flag == 'redo':
        global redo
        redo = True

    if data_nm == '':
        gen_qsubs()
        return

    if data_nm == 'plot':
        plot()

    else:
        load_statistics(data_nm)

    return


if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(data_nm=sys.argv[1])
    elif len(sys.argv) == 3:
        main(data_nm=sys.argv[1], redo_flag=sys.argv[2])
    else:
        main()
