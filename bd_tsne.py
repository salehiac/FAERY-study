import json
from sklearn.decomposition import PCA


from utils_misc import get_path
from t_sne.utils_tsne import *
from t_sne.class_solver_extractor import SolverExtractor


with open(get_path(default="tsne.json"), 'r') as f:
    params = json.load(f)


title_bland = "TSNE on the behavior descriptors of {} sampled solvers\nin Metworld assembly-v2"\
    .format(params["nb_samples"])
title_highlight = title_bland
title_animation = "{} ({}), meta-step={}, inner_step={}"


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

    list_perplex_to_tsne = [
        compute_tsne(
            input_list=extractor.list,
            perplexities=params["perplexities"],
            verbose=True,
            to_val=to_val,
            pca_components=params["pca_components"] if i == 0 and params["param_to_solver"] is True else None
        ) for i, to_val in enumerate(list_to_val)
    ]
    
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
                for algo, color in params["to_follow"].items():
                    tmp_to_follow = {algo:color}

                    plot_follow(

                        to_highlight=tmp_to_follow,
                        
                        save_name=params["save_basename"].format("{}_animated").format(algo),
                        save_path=params["save_path"],

                        base_title=title_animation,
                        
                        meta_steps=params["meta_steps"] if type(params["meta_steps"]) is not str
                            else {"train":extractor.meta_steps["train"]},

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

                # if i == 0:
                #     axs[i].set_title(params["titles_side"][i].format("{}", params["pca_components"]))
                # axs[i].set_title(params["titles_side"][i].format(perplexities[0]))

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

                            **tmp_params
                        )
                        
                        # if k == 0:
                        #     axs[i][k].set_title(params["titles_side"][k].format("{}", params["pca_components"]))
                        # axs[i][k].set_title(params["titles_side"][k].format(perplexities[0]))

                    handles, labels = [], []
                    for i, ax in enumerate(axs):
                        h, l = ax[0].get_legend_handles_labels()
                        handles += h[int(i>0):]
                        labels += l[int(i>0):]
                    
                    fig.legend(handles, labels)

                    plt.savefig("{}/{}.png".format(save_path, save_name))
