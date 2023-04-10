# import inDelphi.neural_network as nn
import inDelphi.neural_network as nn
# import inDelphi.nearest_neighbours as knn
from inDelphi.util import get_data, init_folders, Filenames
def train_model(data_url, out_place):
    for dropout_rate in [0.2, 0.0, 0.1]:
        for seed in [42]:
            out_dir, out_letters, out_dir_params, log_fn = init_folders(out_place)
            filenames = Filenames(out_dir, out_letters, out_dir_params, log_fn)
            master_data = get_data(data_url, log_fn)

            nn_params, nn_2_params = nn.train_and_create(master_data, filenames, num_epochs=100, dropout_rate=dropout_rate, seed_n=seed)

if __name__ == '__main__':
    train_model('pickle_data/inDelphi_counts_and_deletion_features.pkl', './experiments/mshen/prj/mmej_figures/out/d2_model/')