import argparse
import random

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from t_sne.utils_tsne import *

# WILL LOAD ALL THE BDS
# WILL DO TSNE ON THEM

nb_samples = 5000

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TSNE visualization')

    parser.add_argument(
        "--meta_dir",
        type=str,
        help="path to meta-learning directory"
    )

    args = parser.parse_args()

    solvers = get_solvers(args.meta_dir)
    to_compute = random.sample(
        solvers,
        min(nb_samples, len(solvers))
    )

    solvers_bds = [ag._behavior_descr[0] for ag in to_compute] 
    embedding = TSNE(n_components=2).fit_transform(solvers_bds)

    plt.plot(embedding[:, 0], embedding[:, 1], "bo", label="solvers")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel("$e_1$",fontsize=14)
    plt.ylabel("$e_2$",fontsize=14)

    plt.show()

