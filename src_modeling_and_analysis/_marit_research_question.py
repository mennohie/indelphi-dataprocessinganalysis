import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import _predict as predict
import pandas as pd
import pickle as pkl
from inDelphi.util import split_data_set


def get_predicted(dataset):
    predict.init_model()

    rate_model, bp_model, normalizer = predict.init_rate_bp_models()
    dataset = pd.merge(dataset['counts'], dataset['del_features'],
                       left_on=dataset['counts'].index, right_on=dataset['del_features'].index,
                       how="left")
    dataset['Length'] = [int(x[1].split("+")[1]) if x[1].split("+")[1].isdigit() else len(x[1].split("+")[1]) for x in
                         dataset["key_0"]]
    dataset['cutSite'] = [int(x[1].split("+")[0]) + 29 for x in dataset["key_0"]]
    dataset['exp'] = [x[0] for x in dataset["key_0"]]

    # TODO ignores cutsites not compatible with liba, is this correct?
    dataset = dataset[dataset['cutSite'] > 4]
    with open("../data_libprocessing/targets-libA.txt") as f:
        full_dna_exps = []
        for line in f:
            full_dna_exps.append(line.strip("\n"))

    exps = list(set(dataset['exp']))

    exps = exps[0:80]  # select fewer exps for testing purposes

    for i, exp in enumerate(exps):
        fraction = 0
        print(i)
        header_data = list(dataset[dataset["exp"] == exp]["exp"])[0].split("_")[:-1]
        header = ""
        for h in header_data:
            header = header + h + "_"

        header.removesuffix("_")

        exp_substring_dna = exp[-20:]
        matched_dna = list(filter(lambda x: exp_substring_dna in x, full_dna_exps))
        if matched_dna:
            sequence = matched_dna[0]
        else:
            print(f"Experiment {exp} not in libA!")
            continue
        cutsites = dataset[dataset["exp"] == exp]["cutSite"]
        for cutsite in cutsites:
            pred_del_df, pred_all_df, total_phi_score, rate_1bpins = predict.predict_all(sequence, cutsite, rate_model, bp_model, normalizer)
            print('here')

if __name__ == '__main__':
    master_data = pkl.load(open('../pickle_data/inDelphi_counts_and_deletion_features_p4.pkl', 'rb'))
    training_data, test_data = split_data_set(master_data)

    predicted = get_predicted(test_data)