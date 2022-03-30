import argparse
import random
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from t_sne.utils_tsne import *

# WILL LOAD ALL THE BDS
# WILL DO TSNE ON THEM

nb_samples = 5000
perplexities = [25, 50, 75, 100]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TSNE visualization')

    parser.add_argument(
        "--meta_dir",
        type=str,
        help="path to meta-learning directory"
    )

    args = parser.parse_args()

    print("Retrieving files")
    solvers = get_solvers(args.meta_dir)
    nb_to_compute = min(nb_samples, len(solvers))

    solvers_to_compute = random.sample(
        solvers,
        nb_to_compute
    )
    
    # params = get_parameters(args.meta_dir)
    # params_to_compute = random.sample(
    #     params,
    #     min(nb_samples, len(params))
    # )

    fig, axs = plt.subplots(ncols=len(perplexities))
    print("Embedding {} solvers among {} retrieved".format(nb_to_compute, len(solvers)))

    for i, perplexity in enumerate(perplexities):
        solvers_bds = np.array([ag._behavior_descr[0] for ag in solvers_to_compute]).reshape(-1,1)
        solvers_embedding = TSNE(n_components=2, perplexity=perplexity).fit_transform(solvers_bds)

        # print("Embedding parameters")
        # params_arr = np.array([np.array(ag.get_flattened_weights()) for ag in params_to_compute]).reshape(-1,1)
        # params_embedding = TSNE(n_components=2).fit_transform(params_arr)

        print("Plotting")
        
        axs[i].scatter(solvers_embedding[:, 0], solvers_embedding[:, 1], label="solvers", color="blue")
        # axs[1].plot(params_embedding[:, 0], params_embedding[:, 1], "bo", label="parameters")

        axs[i].set_title("Perplexity: {}".format(perplexity))

    plt.suptitle("TSNE on the behavior descriptors of {} sampled solvers\nin Metworld assembly-v2".format(nb_to_compute))
    plt.show()

