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
import shap

def get_pred_function(nn_params):
    def get_pred(sample):
        res = []
        for sequence in sample:
            mh_len = sequence[0]
            gc_frac = sequence[1]
            del_len = sequence[2]
            inp = np.array([mh_len, gc_frac]).T
            del_lens = np.array(del_len).T
            mh_scores = nn_match_score_function(nn_params, inp)
            js = np.array(del_lens)
            unnormalized_fq = np.exp(mh_scores - 0.50 * js)
            # unnormalized_fq = np.exp(mh_scores)
            mh_phi_total = np.sum(unnormalized_fq, dtype=np.float64)
            res.append(mh_phi_total)
        return np.asarray(res)
    return get_pred

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
    return ans

def get_param(run_iter='aaf', param_iter='aae'):
    model_out_dir = '/cluster/mshen/prj/mmej_figures/out/d2_model/'
    param_fold = model_out_dir + '%s/parameters/' % run_iter
    nn_params = pickle.load(open("./" + param_fold + '%s_nn.pkl' % param_iter, "rb"))
    nn2_params = pickle.load(open("./" + param_fold + '%s_nn2.pkl' % param_iter, "rb"))

    return nn_params, nn2_params

def compute_shap_values():
    INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = get_data()
    nn_params, nn2_params = get_param()
    np.concatenate(INP_test).ravel()
    del_feat = np.concatenate(DEL_LENS_train).ravel()
    mh_feat = np.concatenate(INP_train).ravel().reshape((len(del_feat), 2))
    samples = np.c_[mh_feat, del_feat]
    del_feat_test = np.concatenate(DEL_LENS_test).ravel()
    mh_feat_test = np.concatenate(INP_test).ravel().reshape((len(del_feat_test), 2))
    samples_test = np.c_[mh_feat_test, del_feat_test]

    explainer = shap.Explainer(get_pred_function(nn_params), samples)
    shap_values = explainer(samples_test)

    out_folder = 'data/shap/scalar'
    pickle.dump(samples, open(out_folder + '/background_nn_1.pkl', 'wb'))
    pickle.dump(samples_test, open(out_folder + '/test_samples_nn_1.pkl', 'wb'))
    pickle.dump(shap_values, open(out_folder + '/shap_values_nn_1.pkl', 'wb'))

if __name__ == '__main__':

    # compute_shap_values()
    #
    out_folder = 'data/shap/normal/'

    shap_values_one = pickle.load(open(out_folder + 'shap_values_nn_1.pkl', 'rb'))
    shap_values_one.feature_names = ['MH len', 'GC frac', 'DEL len']
    shap.plots.scatter(shap_values_one[:,'DEL len'])
    plt.tight_layout()
    plt.savefig("figures/shap/normal/scatter_2.png")

