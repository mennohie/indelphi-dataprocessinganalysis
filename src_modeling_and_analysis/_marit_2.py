from d2_model import nn_match_score_function
import autograd.numpy as np
import autograd.numpy.random as npr
import pickle
from inDelphi.util import split_data_set
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def get_predict(nn_params, nn2_params, mh_NN_inp, obs_freqs, del_lens):
    obs_freqs_list = []
    predicted_freq_list = []
    for idx in tqdm.tqdm(range(len(mh_NN_inp)), desc="#GRNA's"):

        ##
        # MH-based deletion frequencies
        ##
        mh_scores = nn_match_score_function(nn_params, mh_NN_inp[idx])
        Js = np.array(del_lens[idx])
        unnormalized_fq = np.exp(mh_scores - 0.25 * Js)

        # Add MH-less contribution at full MH deletion lengths
        mh_vector = mh_NN_inp[idx].T[0]
        mhfull_contribution = np.zeros(mh_vector.shape)
        for jdx in range(len(mh_vector)):
            if del_lens[idx][jdx] == mh_vector[jdx]:
                dl = del_lens[idx][jdx]
                mhless_score = nn_match_score_function(nn2_params, np.array(dl))
                mhless_score = np.exp(mhless_score - 0.25 * dl)
                mask = np.concatenate(
                    [np.zeros(jdx, ), np.ones(1, ) * mhless_score, np.zeros(len(mh_vector) - jdx - 1, )])
                mhfull_contribution = mhfull_contribution + mask
        unnormalized_fq = unnormalized_fq + mhfull_contribution
        normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))
        obs_freqs_list.append(np.divide(obs_freqs[idx], np.sum(obs_freqs[idx])))
        predicted_freq_list.append(normalized_fq)

    return obs_freqs_list, predicted_freq_list

def get_param(run_iter='aab', param_iter='aae'):
    model_out_dir = '/cluster/mshen/prj/mmej_figures/out/d2_model/'
    param_fold = model_out_dir + '%s/parameters/' % run_iter
    nn_params = pickle.load(open("./" + param_fold + '%s_nn.pkl' % param_iter, "rb"))
    nn2_params = pickle.load(open("./" + param_fold + '%s_nn2.pkl' % param_iter, "rb"))

    return nn_params, nn2_params

def get_data():
    master_data = pickle.load(open("../pickle_data/inDelphi_counts_and_deletion_features.pkl", "rb"))
    training_data, test_data = split_data_set(master_data)

    '''
  Unpack data from e11_dataset
  '''

    res = pd.merge(training_data['counts'], training_data['del_features'], left_on=training_data['counts'].index,
                   right_on=training_data['del_features'].index)
    res[['sample', 'offset']] = pd.DataFrame(res['key_0'].tolist(), index=res.index)
    mh_lens = []
    gc_fracs = []
    del_lens = []
    exps = []
    freqs = []
    dl_freqs = []
    for group in res.groupby("sample"):
        mh_lens.append(group[1]['homologyLength'].values)
        gc_fracs.append(group[1]['homologyGCContent'].values)
        del_lens.append(group[1]['Size'].values)
        exps.append(group[1]['key_0'].values)
        freqs.append(group[1]['countEvents'].values)
        dl_freqs.append(group[1]['fraction'].values)

    INP = []
    for mhl, gcf in zip(mh_lens, gc_fracs):
        inp_point = np.array([mhl, gcf]).T  # N * 2
        INP.append(inp_point)
    INP = np.array(INP, dtype=object)  # 2000 * N * 2
    # Neural network considers each N * 2 input, transforming it into N * 1 output.
    OBS = np.array(freqs, dtype=object)
    OBS2 = np.array(dl_freqs, dtype=object)
    NAMES = np.array([str(s) for s in exps])
    DEL_LENS = np.array(del_lens, dtype=object)

    seed = npr.RandomState(1)
    ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size=0.15, random_state=seed)
    INP_train, INP_test, OBS_FREQS_train, OBS_FREQS_test, OBS_FRAC_train, OBS_FRAC_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans


    return INP_test, OBS_FREQS_test, DEL_LENS_test

if __name__ == '__main__':
    nn_params, nn2_params = get_param()
    inp_test, obs_test, del_len = get_data()

    obs, pred = get_predict(nn_params, nn2_params, inp_test, obs_test, del_len)


    for i in range(len(obs)):
        plt.clf()
        plt.plot([0, 1], linestyle='dashed', c='black')
        plt.scatter(obs[i], pred[i])
        sns.regplot(x=obs[i], y=pred[i], scatter=False, color='red')
        plt.savefig("figures/scatter/whole_"+ str(i) + " .png")

        if i == 5:
            break



