import json
import matplotlib.animation as animation

from utils_tsne import *
from utils_extract import get_path, create_path
from class_solver_extractor import SolverExtractor


with open(get_path(default="params_tsne.json"), 'r') as f:
    params = json.load(f)


params["load_directory"] = params["load_directory"].format(params["env"])
params["save_path"] = params["save_path"].format(params["env"])
create_path(params["save_path"])

title_bland = "TSNE on {} sampled solvers\nof Metaworld {}"\
    .format(params["nb_samples"], params["env"])
title_highlight = title_bland
title_animation = "{} ({}), meta-step={}, inner_step={}"
title_cluster = "TSNE animation of the mapping between parameters and behavior space\n" \
    + "on {} sampled solvers of Metaworld {}".format(params["nb_samples"], params["env"])


if __name__ == "__main__":

    print("Retrieving files in {}".format(params["load_directory"]))
    extractor = SolverExtractor(load_path=params["load_directory"], max_samples=params["nb_samples"])

    print("Computing TSNEs")
    list_to_val = [
        lambda x: np.array(x.get_flattened_weights()),
        lambda x: x._behavior_descr[0]
    ]

    if params["param_to_solver"] is False:
        list_to_val = [list_to_val[int(params["param_or_solver"] in "solvers")]]

    list_perplex_to_tsne = []
    list_perplex_to_tsne_obj = []
    for i, to_val in enumerate(list_to_val):

        tsne_obj, tsne_emb = compute_tsne(   
            input_list=extractor.list,
            perplexities=params["perplexities"],
            verbose=True,
            to_val=to_val,
            pca_components=params["pca_components"] if i == 0 and params["param_to_solver"] is True else None
        )

        list_perplex_to_tsne_obj.append(tsne_obj)
        list_perplex_to_tsne.append(tsne_emb)

    print("Unpacking TSNEs")
    list_perplex_to_extractor = [
        {
        p: SolverExtractor(solvers_dict=extractor.unpack(list_perplex_to_tsne[i][p]))
        for p in list_perplex_to_tsne[i].keys()
        } for i in range(len(list_to_val))
    ]

    if params["param_to_solver"] is False:
        for perplex_to_extractor in list_perplex_to_extractor:
            params_data = {
                "perplexities": params["perplexities"],
                "perplex_to_extractor": perplex_to_extractor,
            }

            if params["bland"] is True:
                print("Plotting")
                plot_highlight(
                    base_label="prior_population",
                    to_highlight={},

                    title=title_bland,
                    save_path=params["save_path"],
                    save_name=params["save_basename"].format('bland'),

                    **params_data,
                    **params["params_plot"],
                )

            if params["highlight"] is True:
                print("Plotting highlight")
                tmp_params = {
                    'title': title_highlight,
                    'save_name': params["save_basename"].format("highlighted"),

                    **params_data,
                    **params["params_plot"],
                }

                save_path = params["save_path"]
                save_name = params["save_basename"].format("highlighted")

                if params["is_sequential"] is True:

                    plot_highlight(
                        base_label="solvers",
                        to_highlight=params["to_highlight"],
                        save_path=save_path,
                        legend=True,
                        **tmp_params
                    )

                else:

                    box_size = params["params_plot"]["box_size"]
                    perplexities = params["perplexities"]
                    to_highlight = params["to_highlight"]
                    
                    fig, axs = plt.subplots(
                        figsize=(box_size * len(perplexities), box_size * len(to_highlight)),
                        nrows=len(to_highlight), ncols=len(perplexities)
                    )

                    fig.tight_layout(rect=[0, 0, 1, .90])

                    if len(perplexities) == 1:
                        axs = [[axs[k]] for k in range(len(axs))]

                    for i, tmp_highlight in enumerate(to_highlight.items()):

                        plot_highlight(
                            fig=fig, axs=axs[i],
                            
                            base_label="solvers",

                            to_highlight={tmp_highlight[0]:tmp_highlight[1]},
                            save_path=None,

                            **tmp_params
                        )

                    handles, labels = [], []
                    for i, ax in enumerate(axs):
                        h, l = ax[0].get_legend_handles_labels()
                        handles += h[int(i>0):]
                        labels += l[int(i>0):]
                    
                    fig.legend(handles, labels)

                    plt.savefig("{}/{}.png".format(save_path, save_name))


            if params["follow"] is True:
                print("Computing animation")
                params_data["list_perplex_to_extractor"] = [params_data["perplex_to_extractor"]]
                del params_data["perplex_to_extractor"]

                for algo, color in params["to_follow"].items():
                    tmp_to_follow = {algo:color}

                    plot_follow(

                        to_highlight=tmp_to_follow,
                        
                        save_name=params["save_basename"].format("{}_animated").format(algo),
                        save_path=params["save_path"],

                        base_title=title_animation,
                        
                        meta_steps=params["meta_steps"],

                        **params_data,
                        **params["params_plot"],
                        **params["params_anim"],
                    )
    else:

        save_path = params["save_path"]
        save_name = params["save_basename"]

        params_data = {
            "perplexities": [params["perplexities"][0]],
        }

        box_size = params["params_plot"]["box_size"]
        perplexities = params_data["perplexities"]

        if params["bland"] is True:
            print("Plotting")

            fig, axs = plt.subplots(
                    figsize=(box_size * len(list_perplex_to_extractor), box_size),
                    nrows=1, ncols=len(list_perplex_to_extractor)
                )
            
            fig.tight_layout(rect=[0, 0, 1, .90])
            
            for i, perplex_to_extractor in enumerate(list_perplex_to_extractor):
                
                plot_highlight(
                    fig=fig, axs=[axs[i]],

                    base_label="prior_population",
                    to_highlight={},

                    title=title_bland,
                    save_path=None,

                    perplex_to_extractor=perplex_to_extractor,

                    **params_data,
                    **params["params_plot"],
                )

                if i == 0:
                    axs[i].set_title(params["titles_side"][i].format(perplexities[0], params["pca_components"]))
                else:
                    axs[i].set_title(params["titles_side"][i].format(perplexities[0]))

            handles, labels = [], []
            for i, ax in enumerate(axs):
                h, l = ax.get_legend_handles_labels()
                handles += h[int(i>0):]
                labels += l[int(i>0):]
            
            fig.legend(handles, labels)

            plt.savefig("{}/{}.png".format(save_path, save_name.format("bland")))
        
        if params["highlight"] is True:
                print("Plotting highlight")
                
                tmp_params = {
                    'title': title_highlight,
                    'save_name': params["save_basename"].format("highlighted"),

                    **params_data,
                    **params["params_plot"],
                }

                to_highlight = params["to_highlight"]

                fig, axs = plt.subplots(
                    figsize=(box_size * len(list_perplex_to_extractor), box_size * len(to_highlight)),
                    nrows=len(to_highlight), ncols=len(list_perplex_to_extractor)
                )

                fig.tight_layout(rect=[0, 0, 1, .90])

                for i, tmp_highlight in enumerate(to_highlight.items()):
                    for k, perplex_to_extractor in enumerate(list_perplex_to_extractor):

                        plot_highlight(
                            fig=fig, axs=[axs[i][k]],
                            
                            base_label="solvers",

                            to_highlight={tmp_highlight[0]:tmp_highlight[1]},
                            save_path=None,

                            perplex_to_extractor=perplex_to_extractor,

                            subtitle=False,

                            **tmp_params
                        )
                        
                handles, labels = [], []
                for i, ax in enumerate(axs):
                    h, l = ax[0].get_legend_handles_labels()
                    handles += h[int(i>0):]
                    labels += l[int(i>0):]
                
                fig.legend(handles, labels)

                axs[0][0].set_title(params["titles_side"][0].format(perplexities[0], params["pca_components"]))
                axs[0][1].set_title(params["titles_side"][1].format(perplexities[0]))

                plt.savefig("{}/{}.png".format(save_path, tmp_params["save_name"]))

        if params["follow"] is True:
            print("Computing animation")

            for algo, color in params["to_follow"].items():
                tmp_to_follow = {algo:color}

                fig, axs = plt.subplots(
                figsize=(box_size * len(list_perplex_to_extractor), box_size),
                nrows=1, ncols=len(list_perplex_to_extractor)
                )

                for k in range(len(list_perplex_to_extractor)):
                    if k == 0:
                        axs[k].set_title(params["titles_side"][k].format(perplexities[0], params["pca_components"]))
                    else:
                        axs[k].set_title(params["titles_side"][k].format(perplexities[0]))

                save_name=params["save_basename"].format("{}_animated").format(algo)
                save_path=params["save_path"]

                movie_writer = animation.FFMpegWriter(fps=params["params_anim"]["fps"])
                movie_writer.setup(fig, "{}/{}.mp4".format(save_path, save_name), dpi=params["params_anim"]["dpi"])
                
                plot_follow(
                    fig=fig, axs=[axs],
                    
                    to_highlight=tmp_to_follow,
                    list_perplex_to_extractor=list_perplex_to_extractor,
                    
                    movie_writer=movie_writer,

                    base_title=title_animation,
                    subtitle=False,
                    
                    meta_steps=params["meta_steps"],

                    **params_data,
                    **params["params_plot"],
                    **params["params_anim"],
                )

                print("Wrapping up..", end='\r')
                movie_writer.finish()
                print()
                print("Done")
        
        if params["clustering"] is True:
            print("Computing clustering")

            fig, axs = plt.subplots(
                figsize=(box_size * len(list_perplex_to_extractor), box_size),
                nrows=1, ncols=len(list_perplex_to_extractor)
            )

            fig.suptitle(title_cluster)

            for k in range(len(list_perplex_to_extractor)):
                if k == 0:
                    axs[k].set_title(params["titles_side"][k].format(perplexities[0], params["pca_components"]))
                else:
                    axs[k].set_title(params["titles_side"][k].format(perplexities[0]))

            save_name=params["save_basename"].format("{}_animated").format("clusters")
            save_path=params["save_path"]

            movie_writer = animation.FFMpegWriter(fps=params["params_anim"]["fps"])
            movie_writer.setup(fig, "{}/{}.mp4".format(save_path, save_name), dpi=params["params_anim"]["dpi"])

            perplex_nb_clusters = {
                p:get_clusters(
                    input_array=list_perplex_to_extractor[0][p].list,
                    quantile=params["cluster_quantile"]
                ) for p in perplexities
            }

            plot_clustering(
                fig=fig, axs=[axs],
                
                list_perplex_to_extractor=list_perplex_to_extractor,
                perplex_to_clusters=perplex_nb_clusters,

                movie_writer=movie_writer,

                **params_data,
                **params["params_plot"],
                **params["params_anim_to_solver"],
            )

            print("Wrapping up..", end='\r')
            movie_writer.finish()
            print()
            print("Done")