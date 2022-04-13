import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import List

from sklearn.manifold import TSNE


def get_files(path:str, basename:str = "solvers", basenames:List[str] = None) -> dict:
    """
    Returns all path's basename content as dict classifying them by algorithm and meta-step
    """

    if basenames == None:   basenames = [basename]

    warning_text = "Please be mindful to provide a path of algorithms' folders with unique names."
    print(warning_text, end='\r')
    print(len(warning_text) * " ", end="\r")

    list_algo_dict = [{} for _ in range(len(basenames))]
    for root, dirs, files in os.walk(path):

        new_root, new_name = os.path.split(root)
        new_root, _ = os.path.split(new_root)
        _, algorithm = os.path.split(new_root)

        try:
            type_run, meta_step = new_name.split('_')

            if "FAERY" not in algorithm:
                raise ValueError
            
            for algo_dict in list_algo_dict:

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
            for i, bn in enumerate(basenames):
                
                if bn in name:
                    
                    with open(os.path.join(root, name), 'rb') as f:
                        content = []
                        while True:
                            try:
                                tmp = pickle.load(f)
                                content += tmp
                            except:
                                break
                        
                        list_algo_dict[i][algorithm][type_run][meta_step][name[len(bn)+1:-4]] = content
                            
    return list_algo_dict


def order_str_int(input_set) -> list:
    """
    Orders a set of str numbers
    """

    ordered = sorted(list(map(int, input_set)))
    return list(map(str, ordered))


def compute_tsne(input_list:list, perplexities=[25,50,75,100], verbose=True, max_samples=5000,
    to_val=[lambda x: x._behavior_descr[0]]) -> dict:
    """
    Returns the computed tsne of the input list for all given perplexities
    to_extract: "bd" or "param"
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

        solver_points = np.array([to_val(ag) for ag in solvers_to_compute]).reshape(-1,1)
        solvers_embedding = TSNE(n_components=2, perplexity=perplexity).fit_transform(solver_points)
        perplexity_to_embedding[perplexity] = solvers_embedding
    
    return perplexity_to_embedding  


def plot_highlight(
    perplexities, perplex_to_extractor,
    fig=None, axs=None, legend=False,

    base_label="all agents", base_color="blue", base_alpha=.25,
    prior_pop_label="prior population", prior_pop_color="red",
    solvers_label="solvers", solvers_alpha=.5,

    to_highlight={},  box_size=12, marker='o',
    title="", save_path=None, save_name=""):
    """
    Plots the TSNEs at different perplexities and highlights the given algorithms
    """

    if fig is None or axs is None:
        fig, axs = plt.subplots(
            figsize=(box_size * len(perplexities), box_size),
            ncols=len(perplexities)
        )

        fig.tight_layout(rect=[0, 0, 1, .90])

        if len(perplexities) == 1:
            axs = [axs]
    
    for i, perplexity in enumerate(perplexities):
        extractor = perplex_to_extractor[perplexity]
        axs[i].scatter(np.array(extractor.list)[:, 0], np.array(extractor.list)[:, 1],
            label=base_label, color=base_color, alpha=base_alpha, marker=marker)

        for algo, color in to_highlight.items():
            points = extractor.get_params(extractor.find_algorithm(algo))

            # Check if points come from parameters dict
            if len(points) == 2:
                prior_pop = np.array(points[0])
                solvers = np.array(points[1])
                
                base_name = "{} {}".format(algo, "{}")

                axs[i].scatter(prior_pop[:, 0], prior_pop[:, 1],
                    label=base_name.format(prior_pop_label), color=prior_pop_color, marker=marker)
                
                axs[i].scatter(solvers[:, 0], solvers[:, 1],
                    label=base_name.format(solvers_label), color=prior_pop_color, alpha=solvers_alpha, marker=marker)
            
            else:
                axs[i].scatter(np.array(points)[:, 0], np.array(points)[:, 1],
                    label=algo, color=color, marker=marker)
        
        axs[i].set_title("Perplexity: {}".format(perplexity))
    
    if legend is True:
        fig.legend(*axs[0].get_legend_handles_labels())

    plt.suptitle(title)

    if save_path is not None:
        plt.savefig("{}/{}.png".format(save_path, save_name))


def plot_follow(
    perplexities, perplex_to_extractor,
    to_highlight, types_run, meta_steps, animate_inner=True,
    fig=None, axs=None, movie_writer=None, legend=False,
    background_alpha=.25, inner_alpha=.5,
    base_color="blue", box_size=12, marker='o',
    base_title="{} {} {} {}", save_path=None, save_name="",
    type_writer=animation.PillowWriter, fps=30, dpi=100, time_pause=1.5):
    """
    Animates the TSNE plots for given meta_steps and inner_steps
    """

    if fig is None or axs is None or movie_writer is None:
        fig, axs = plt.subplots(
            figsize=(box_size * len(perplexities), box_size),
            ncols=len(perplexities)
        )

        fig.tight_layout(rect=[0, 0, 1, .95])
        
        if len(perplexities) == 1:
            axs = [axs]

        movie_writer = type_writer(fps=fps)
        movie_writer.setup(fig, "{}/{}.gif".format(save_path, save_name), dpi=dpi)

    # Plotting the background solvers
    for k, perplexity in enumerate(perplexities):
        points = np.array(perplex_to_extractor[perplexity].list)
        axs[k].scatter(points[:, 0], points[:, 1],
            label="solvers", color=base_color, marker=marker, alpha=background_alpha)
        axs[k].set_title("Perplexity: {}".format(perplexity))

    # Extractors all have the same structure, just not the same values
    basic_extractor = list(perplex_to_extractor.values())[0]
    
    # Animation
    for i1, (algo, color) in enumerate(to_highlight.items()):

        objects = [
            (
                axs[k].plot([], [], label=algo, color=color, marker=marker, alpha=1, linestyle='None')[0],
                axs[k].plot([], [], label=algo, color=color, marker=marker, alpha=inner_alpha, linestyle='None')[0],
            )
            for k in range(len(perplexities))
        ]

        algorithm = basic_extractor.find_algorithm(algo)

        for i2, type_run in enumerate(types_run):
            for i3, meta_step in enumerate(meta_steps[type_run]):
                inner_steps = order_str_int(basic_extractor.solvers_dict[algorithm][type_run][meta_step].keys())
                iterate_steps = inner_steps if animate_inner is True else [inner_steps[0]]

                all_points = {p:[] for p in perplexities}
                for i4, inner_step in enumerate(iterate_steps):
                    for i5, perplexity in enumerate(perplexities):

                        print(
                            "Algorithm {}/{} Type_run {}/{} Meta_step {}/{} Inner_step {}/{} Perplexity {}/{}"\
                                .format(
                                    i1+1, len(to_highlight),
                                    i2+1, len(types_run),
                                    i3+1, len(meta_steps[type_run]),
                                    i4+1, len(iterate_steps),
                                    i5+1, len(perplexities)
                                ),
                            end="\r"
                        )

                        points = np.array(perplex_to_extractor[perplexity].get_params(
                            input_algorithm=algo,
                            input_type=type_run,
                            input_meta=meta_step,
                            input_step=inner_step
                        ))

                        if i4 == 0:
                            objects[i5][0].set_data(points[:,0], points[:,1])

                            if legend is True and i5 == 0 and i3 == 0 and i2 == 0:
                                fig.legend(*axs[i5].get_legend_handles_labels())

                        else:
                            
                            all_points[perplexity] += [p for p in points]
    
                            objects[i5][1].set_data([p[0] for p in all_points[perplexity]],
                                [p[1] for p in all_points[perplexity]])       

                    plt.suptitle(base_title.format(algo, type_run, meta_step, inner_step))
                    movie_writer.grab_frame()

                    if i4 == len(inner_steps) - 1:
                        for _ in range(int(fps * time_pause)):
                            movie_writer.grab_frame()

                else:
                    # Erasing the previous pop for the next meta-step
                    for i5, perplexity in enumerate(perplexities):
                        objects[i5][1].set_data([], [])
    
    print("Wrapping up..", end='\r')
    movie_writer.finish()
    print()
    print("Done")
