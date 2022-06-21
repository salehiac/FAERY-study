import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import List

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth


def get_files(path:str, basename:str = "solvers", basenames:List[str] = None, max_samples:int = 5000) -> dict:
    """
    Returns all path's basename content as dict classifying them by algorithm and meta-step
    """

    if basenames == None:   basenames = [basename]

    warning_text = "Please be mindful to provide a path of algorithms' folders with unique names."
    print(warning_text, end='\r')
    print(len(warning_text) * " ", end="\r")

    nb_sampled = 0
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
            print(algorithm, type_run, meta_step, nb_sampled, end="\r")
            if nb_sampled >= max_samples:
                break

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
                        nb_sampled += len(content)

    print("Sampled {} individuals".format(nb_sampled))
    return list_algo_dict


def order_str_int(input_set) -> list:
    """
    Orders a set of str numbers
    """

    ordered = sorted(list(map(int, input_set)))
    return list(map(str, ordered))


def compute_tsne(input_list:list, perplexities=[25,50,75,100], verbose=True,
    to_val=lambda x: x._behavior_descr[0], pca_components=None) -> dict:
    """
    Returns the computed tsne of the input list for all given perplexities
    to_extract: "bd" or "param"
    """

    solvers_to_compute = input_list[:]
    solver_points = np.array([to_val(ag).reshape(1,-1)[0] for ag in solvers_to_compute])

    if pca_components is not None:
        if type(pca_components) != int:
            raise ValueError("Please enter a number for the pca_components argument")
        
        print("Computung PCA with {} components..".format(pca_components), end='\r')
        pca = PCA(n_components=pca_components)
        solver_points = pca.fit_transform(solver_points)

    perplexity_to_tsne = {}
    perplexity_to_embedding = {}
    for i, perplexity in enumerate(perplexities):

        if verbose is True:
            print("Computing TSNE for {} perplexity..".format(perplexity), end='\r')
        
        perplexity_to_tsne[perplexity] = TSNE(n_components=2, perplexity=perplexity, n_jobs=-1) # useful to gather posterior info
        solvers_embedding = perplexity_to_tsne[perplexity].fit_transform(solver_points)
        perplexity_to_embedding[perplexity] = solvers_embedding
    
    return perplexity_to_tsne, perplexity_to_embedding  


def plot_highlight(
    perplexities, perplex_to_extractor,
    fig=None, axs=None, legend=False,

    base_label="all agents", base_color="blue", base_alpha=.25,
    prior_pop_label="prior population", prior_pop_color="red",
    solvers_label="solvers", solvers_alpha=.5,

    subtitle=True,

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
        
        if subtitle is True:    axs[i].set_title("Perplexity: {}".format(perplexity))
    
    if legend is True:
        fig.legend(*axs[0].get_legend_handles_labels())

    plt.suptitle(title)

    if save_path is not None:
        plt.savefig("{}/{}.png".format(save_path, save_name))


def plot_follow(
    perplexities, list_perplex_to_extractor,
    to_highlight, types_run, meta_steps, animate_inner=True,
    fig=None, axs=None, movie_writer=None, legend=False,
    background_alpha=.25, inner_alpha=.5,
    base_color="blue", box_size=12, marker='o',
    base_title="{} {} {} {}", save_path=None, save_name="",
    subtitle=True,
    type_writer=animation.FFMpegWriter, fps=30, dpi=100, time_pause=1.5):
    """
    Animates the TSNE plots for given meta_steps and inner_steps
    """

    if fig is None or axs is None:
        fig, axs = plt.subplots(
            figsize=(box_size * len(perplexities), box_size),
            ncols=len(perplexities)
        )

        fig.tight_layout(rect=[0, 0, 1, .95])
        
        if len(perplexities) == 1:
            axs = [[axs]]

    movie_was_none = movie_writer is None
    if  movie_was_none:
        movie_writer = type_writer(fps=fps)
        movie_writer.setup(fig, "{}/{}.mp4".format(save_path, save_name), dpi=dpi)

    # Plotting the background solvers
    for k, perplexity in enumerate(perplexities):
        for i, perplex_to_extractor in enumerate(list_perplex_to_extractor):
            points = np.array(perplex_to_extractor[perplexity].list)
            axs[k][i].scatter(points[:, 0], points[:, 1],
                label="solvers", color=base_color, marker=marker, alpha=background_alpha)
            if subtitle is True: axs[k][i].set_title("Perplexity: {}".format(perplexity))

    # Extractors all have the same structure, just not the same values
    basic_extractor = list(list_perplex_to_extractor[0].values())[0]
    
    # Animation
    for i1, (algo, color) in enumerate(to_highlight.items()):

        objects = [
            [
                (
                    axs[k][i].plot([], [], label=algo, color=color, marker=marker, alpha=1, linestyle='None')[0],
                    axs[k][i].plot([], [], label=algo, color=color, marker=marker, alpha=inner_alpha, linestyle='None')[0],
                )
                for k in range(len(perplexities))
            ]
            for i in range(len(list_perplex_to_extractor))
        ]

        algorithm = basic_extractor.find_algorithm(algo)

        for i2, type_run in enumerate(types_run):
            meta_step_counter = 0
            for i3, meta_step in enumerate(basic_extractor.meta_steps[type_run][algorithm]):
                
                if meta_steps != "all" and meta_step not in meta_steps:
                    continue
                meta_step_counter += 1
                
                inner_steps = order_str_int(basic_extractor.solvers_dict[algorithm][type_run][meta_step].keys())
                iterate_steps = inner_steps if animate_inner is True else [inner_steps[0]]

                all_points = [{p:[] for p in perplexities} for k in range(len(list_perplex_to_extractor))]
                for i4, inner_step in enumerate(iterate_steps):
                    for i5, perplexity in enumerate(perplexities):

                        print(
                            "Algorithm {}/{} Type_run {}/{} Meta_step {}/{} Inner_step {}/{} Perplexity {}/{}"\
                                .format(
                                    i1+1, len(to_highlight),
                                    i2+1, len(types_run),
                                    meta_step_counter, len(meta_steps),
                                    i4+1, len(iterate_steps),
                                    i5+1, len(perplexities)
                                ),
                            end="\r"
                        )

                        for k, perplex_to_extractor in enumerate(list_perplex_to_extractor):
                            points = np.array(perplex_to_extractor[perplexity].get_params(
                                input_algorithm=algo,
                                input_type=type_run,
                                input_meta=meta_step,
                                input_step=inner_step
                            ))

                            if i4 == 0:
                                objects[k][i5][0].set_data(points[:,0], points[:,1])

                                if legend is True and i5 == 0 and i3 == 0 and i2 == 0:
                                    fig.legend(*axs[i5].get_legend_handles_labels())

                            else:
                                
                                all_points[k][perplexity] += [p for p in points]
        
                                objects[k][i5][1].set_data([p[0] for p in all_points[k][perplexity]],
                                    [p[1] for p in all_points[k][perplexity]])       

                    plt.suptitle(base_title.format(algo, type_run, meta_step, inner_step))
                    movie_writer.grab_frame()

                    if i4 == len(inner_steps) - 1:
                        for _ in range(int(fps * time_pause)):
                            movie_writer.grab_frame()

                else:
                    # Erasing the previous pop for the next meta-step
                    for i5, perplexity in enumerate(perplexities):
                        for k in range(len(list_perplex_to_extractor)):
                            objects[k][i5][1].set_data([], [])
    
    if movie_was_none:
        print("Wrapping up..", end='\r')
        movie_writer.finish()
        print()
        print("Done")


def distance_euclidian(a, b):
    assert len(a) == len(b), "Objects don't have same length : {} and {}".format(len(a), len(b))
    return np.sqrt(sum([(a[i] - b[i])**2 for i in range(len(a))]))


def k_means(points, centers_clusters, min_distance=float('-inf'), distance=distance_euclidian):
    """
    Applies K-means algorithm to a given set of points, returns clusters as lists of point index
    """

    clusters = [
        [c, []]
        for c in centers_clusters
    ] # [(center, point index)]

    point_to_cluster = {
        ip:None for ip in range(len(points))
    }

    convergence = False
    while convergence is False:
        convergence = True

        for i, point in enumerate(points):
            distances = [distance(point, cluster[0]) for cluster in clusters]

            best_distance = distances.index(min(distances))

            if point_to_cluster[i] is not None:
                clusters[point_to_cluster[i]][-1].remove(i)
            
            convergence = convergence and \
                ((best_distance == point_to_cluster[i]) or (distances[best_distance] <= min_distance))

            point_to_cluster[i] = best_distance
            clusters[best_distance][-1].append(i)
        
        for cluster in clusters:
            if len(cluster) >= 0:

                s = [0 for _ in range(len(cluster[0]))]
                for p in cluster[-1]:
                    for k, ip in enumerate(points[p]):
                        s[k] += ip
                
                s = [s[i] / len(cluster[0]) for i in range(len(s))]
                cluster[0] = s
    
    return tuple([c[-1] for c in clusters])


def get_clusters(input_array, quantile=.005):
    """
    Returns the number of estimated clusters in array
    """

    bandwidth = estimate_bandwidth(input_array, quantile=quantile, n_samples=len(input_array))

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(input_array)

    return ms.cluster_centers_, len(np.unique(ms.labels_))


def plot_clustering(
    perplexities, list_perplex_to_extractor,
    perplex_to_clusters,
    fig=None, axs=None, movie_writer=None,
    color="red",
    animate_inner=True,
    background_alpha=.25,
    base_color="blue", box_size=12, marker='o',
    subtitle=False,
    save_path=None, save_name="",
    type_writer=animation.PillowWriter, fps=30, dpi=100, time_pause=1.5):
    
    """
    Animates a plot, highlights the parameters space cluster by cluster
    """

    if fig is None or axs is None:
        fig, axs = plt.subplots(
            figsize=(box_size * len(perplexities), box_size),
            ncols=len(perplexities)
        )

        fig.tight_layout(rect=[0, 0, 1, .95])
        
        if len(perplexities) == 1:
            axs = [[axs]]

    if  movie_writer is None:
        movie_writer = type_writer(fps=fps)
        movie_writer.setup(fig, "{}/{}.gif".format(save_path, save_name), dpi=dpi)

    for k, perplexity in enumerate(perplexities):

        # Plotting the background solvers
        for i, perplex_to_extractor in enumerate(list_perplex_to_extractor):
            points = np.array(perplex_to_extractor[perplexity].list)
            axs[k][i].scatter(points[:, 0], points[:, 1],
                label="solvers", color=base_color, marker=marker, alpha=background_alpha)
            if subtitle is True:    axs[k][i].set_title("Perplexity: {}".format(perplexity))
    
        # Retrieve the clusters from the parameters space
        perplex_to_clusters = {
            perplexity:k_means(
                points=list_perplex_to_extractor[0][perplexity].list,
                centers_clusters=perplex_to_clusters[perplexity][0],
                min_distance=0.01
            )
            for perplexity in perplexities
        }

    # Animation
    objects = [
        [
            axs[k][i].plot([], [], color=color, marker=marker, alpha=1, linestyle='None')[0]
            for k in range(len(perplexities))
        ]
        for i in range(len(list_perplex_to_extractor))
    ]
    
    for i1, perplexity in enumerate(perplexities):
        for i2, cluster in enumerate(perplex_to_clusters[perplexity]):

            all_points = [[] for _ in range(len(list_perplex_to_extractor))]

            for i3, ipoints in enumerate(cluster):
                for i4, perplex_to_extractor in enumerate(list_perplex_to_extractor):

                    print("Perplexity {}/{} Cluster {}/{} Point {}/{} Graph {}/{}"\
                        .format(i1,len(perplexities), i2, len(perplex_to_clusters[perplexity]),
                        i3, len(cluster), i4, len(list_perplex_to_extractor)),
                        end="\r")

                    all_points[i4].append(perplex_to_extractor[perplexity].list[ipoints])

                    objects[i4][i1].set_data([point[0] for point in all_points[i4]],
                                             [point[1] for point in all_points[i4]])

                if animate_inner is True:
                    movie_writer.grab_frame()

            for _ in range(int(fps * time_pause)):
                movie_writer.grab_frame()
            
            # Erasing the previous pop for the next meta-step
            for i5, perplexity in enumerate(perplexities):
                for k in range(len(list_perplex_to_extractor)):
                    objects[k][i5].set_data([], [])
    
    if movie_writer is None:
        print("Wrapping up..", end='\r')
        movie_writer.finish()
        print()
        print("Done")