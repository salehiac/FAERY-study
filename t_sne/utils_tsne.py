import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def get_files(path:str, basename:str) -> dict:
    """
    Returns all path's basename content as dict classifying them by algorithm and meta-step
    """

    warning_text = "Please be mindful to provide a path of algorithms' folders with unique names."
    print(warning_text, end='\r')
    print(len(warning_text) * " ", end="\r")

    algo_dict = {}
    for root, dirs, files in os.walk(path):

        new_root, new_name = os.path.split(root)
        new_root, _ = os.path.split(new_root)
        _, algorithm = os.path.split(new_root)

        try:
            type_run, meta_step = new_name.split('_')

            if "FAERY" not in algorithm:
                raise ValueError
                
            if algorithm not in algo_dict.keys():

                algo_dict[algorithm] = {
                    "train": {},
                    "test": {}
                }
            
            if meta_step not in algo_dict[algorithm][type_run].keys():
                algo_dict[algorithm][type_run][meta_step] = {}

        except ValueError:
            # Raised if sub_folder's name is invalid
            continue
        except KeyError:
            # Raised if sub_folder wasn't that of a train or test
            continue

        for name in files:
            if basename in name:
                with open(os.path.join(root, name), 'rb') as f:
                    inner_step = name[len(basename)+1:-4]
                    algo_dict[algorithm][type_run][meta_step][inner_step] = pickle.load(f)

    return algo_dict


def order_str_int(input_set) -> list:
    """
    Orders a set of str numbers
    """

    ordered = sorted(list(map(int, input_set)))
    return list(map(str, ordered))


def compute_tsne(input_list:list, perplexities=[25,50,75,100], verbose=True, max_samples=5000) -> dict:
    """
    Returns the computed tsne of the input list for all given perplexities
    """

    nb_to_compute = min(max_samples, len(input_list))

    solvers_to_compute = random.sample(
        input_list,
        nb_to_compute
    )
    
    if verbose is True:
        print("Sampled {} solvers among {} retrieved".format(nb_to_compute, len(input_list)))

    perplexity_to_embedding = {}
    for i, perplexity in enumerate(perplexities):

        if verbose is True:
            print("Computing for {} perplexity".format(perplexity), end='\r')

        solvers_bds = np.array([ag._behavior_descr[0] for ag in solvers_to_compute]).reshape(-1,1)
        solvers_embedding = TSNE(n_components=2, perplexity=perplexity).fit_transform(solvers_bds)
        perplexity_to_embedding[perplexity] = solvers_embedding
    
    return perplexity_to_embedding  


def plot_highlight(
    perplexities,
    perplex_to_extractor,
    to_highlight={}, base_color="blue", box_size=12, marker='o',
    title="", save_path=None, save_name=""):
    """
    Plots the TSNEs at different perplexities and highlights the given algorithms
    """

    fig, axs = plt.subplots(
        figsize=(box_size * len(perplexities), box_size),
        ncols=len(perplexities)
    )

    if len(perplexities) == 1:
        axs = [axs]
    
    for i, perplexity in enumerate(perplexities):
        extractor = perplex_to_extractor[perplexity]
        axs[i].scatter(np.array(extractor.list)[:, 0], np.array(extractor.list)[:, 1],
            label="solvers", color=base_color, marker=marker)

        for algo, color in to_highlight.items():
            points = extractor.get_algorithm(algo)
            axs[i].scatter(np.array(points)[:, 0], np.array(points)[:, 1],
                label=algo, color=color, marker=marker)
        
        axs[i].set_title("Perplexity: {}".format(perplexity))
    
    fig.legend(*axs[0].get_legend_handles_labels())
    plt.suptitle(title)

    if save_path is None:
        plt.show()
    else:
        plt.savefig("{}/{}.png".format(save_path, save_name))
