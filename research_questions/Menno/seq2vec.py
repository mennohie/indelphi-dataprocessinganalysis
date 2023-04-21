import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Use to initialize, then input in seq2vec
def lib_to_fasta():
    names = []
    with open("../../data_libprocessing/names-libA.txt") as f:
        for line in f:
            names.append(line.strip("\n"))
    grna = []
    with open("../../data_libprocessing/grna-libA.txt") as f:
        for line in f:
            grna.append(line)
    with open("./grna-libA.fasta", "a") as f:
        for i in range(len(names)):
            f.write(f">{names[i]}\n")
            f.write(f"{grna[i]}\n")


def open_embedding_dataset(url="../../data_libprocessing/names-libA.txt",
                           embedding_url='./grna-embedding.tsv'):
    names = []
    with open(url) as f:
        for line in f:
            names.append(line.strip("\n"))
    df = pd.read_csv(embedding_url, sep=",", header=None)
    df["names"] = names
    df = df.set_index('names')
    return df


def plot_pca_variance_ratio(df, n_components=2):
    pca = PCA(n_components=n_components)
    pca.fit(df.T)
    plt.plot(pca.explained_variance_ratio_)
    plt.show()


if __name__ == "__main__":
    lib_to_fasta()
