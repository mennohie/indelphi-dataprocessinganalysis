import pickle as pkl
import autograd.numpy as np
import autograd.numpy.random as npr
from inDelphi.util import split_data_set
import pandas as pd
from sklearn.model_selection import train_test_split
from d2_model import nn_match_score_function
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import _predict as predict
import seaborn as sns
from collections import defaultdict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def observed_dl(data, exps):
    fractions = data['counts']['fraction']
    result = {}

    for i, sequence in enumerate(exps):
        indels = fractions[sequence]
        for pos_indel in indels.index:
            pos, indel = pos_indel.split("+")
            if not indel.isdigit():
                indel = len(indel)

            indel = int(indel)

            fraction = fractions[sequence][pos_indel]
            if fraction != 0.0:
                if sequence in result:
                    if indel in result[sequence]:
                        result[sequence][indel] += fraction
                    else:
                        result[sequence][indel] = fraction
                else:
                    result[sequence] = {indel : fraction}

        # only plot some datapoints for debugging
        # if i == 80000:
        #     break

    return result

def get_exps(dataset):
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
    exps = exps[0:4]

    return exps

def get_deletion_lengths(pred_del_df, del_len):
    for index, row in pred_del_df.iterrows():
        dl = row['Length']
        if dl in del_len:
            del_len[dl] += row['Predicted_Frequency']
        else:
            del_len[dl] = row['Predicted_Frequency']

    return del_len

def get_predicted(dataset):
    predict.init_model()
    all_data = defaultdict(list)
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
    exps = exps[0:10]
    for i, exp in enumerate(exps):
        dl_len = {}
        print("sequence: ", exp)
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

        print("cutsites: ", len(cutsites))
        for cutsite in cutsites:
            pred_del_df, pred_all_df, total_phi_score, rate_1bpins = predict.predict_all(sequence, cutsite, rate_model, bp_model, normalizer)
            dl_len = get_deletion_lengths(pred_del_df, dl_len)

        dl_len = {k: v / sum(dl_len.values()) for k, v in dl_len.items()}
        all_data[exp] = dl_len

    return all_data, exps

def merge_data(obs, pred):
    max_dl = max(max(obs.keys()), max(pred.keys())) + 1
    obs_fracs = []
    pred_fracs = []
    for i in range(max_dl):
        obs_frac = 0
        pred_frac = 0
        if i in obs:
            obs_frac = obs[i]

        if i in pred:
            pred_frac = pred[i]

        obs_fracs.append(obs_frac)
        pred_fracs.append(pred_frac)

    return obs_fracs, pred_fracs



if __name__ == '__main__':
    master_data = pkl.load(open('../pickle_data/inDelphi_counts_and_deletion_features_p4.pkl', 'rb'))
    training_data, test_data = split_data_set(master_data)
    predicted, exps = get_predicted(test_data)

    observed = observed_dl(test_data, exps)
    #
    # for sequence in observed.keys():
    #     plt.clf()
    #     plt.bar(range(len(observed[sequence])), observed[sequence].values(), tick_label=observed[sequence].keys())
    #     plt.show()
    #     plt.savefig("figures/bars/observed/"+ sequence +".png")

    # for sequence in predicted.keys():
    #     plt.clf()
    #     plt.bar(range(len(predicted[sequence])), predicted[sequence].values(), tick_label=predicted[sequence].keys())
    #     plt.show()
    #     plt.savefig("figures/bars/predicted/"+ sequence +".png")

    # plt.clf()
    obs = []
    pred = []
    for sequence in predicted.keys():
        # plt.clf()
        obs_fracs, pred_fracs = merge_data(observed[sequence], predicted[sequence])
        obs.append(obs_fracs)
        pred.append(pred_fracs)

    obs = np.array(obs).flatten()
    pred = np.array(pred).flatten()
    plt.plot([0, 1], linestyle='dashed', c='black')
    corr, p_value = pearsonr(obs, pred)
    plt.title("inDelphi Pearson r = " + "{:.2f}".format(corr))
    plt.scatter(obs, pred)
    sns.regplot(x=obs, y=pred, scatter=False, color='red')

    plt.savefig("figures/scatter/in_nn/result.png")






