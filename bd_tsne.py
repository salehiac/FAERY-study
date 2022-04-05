import json
import matplotlib.animation as animation

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
    extractor = SolverExtractor(load_path=params["load_directory"])

    print("Computing TSNEs")
    perplex_to_tsne = compute_tsne(
        input_list=extractor.list,
        perplexities=params["perplexities"],
        max_samples=params["nb_samples"],
        verbose=True
    )

    print("Unpacking TSNEs")
    perplex_to_extractor = {
        p: SolverExtractor(solvers_dict=extractor.unpack(perplex_to_tsne[p]))
        for p in perplex_to_tsne.keys()
    }

    params_data = {
        "perplexities": params["perplexities"],
        "perplex_to_extractor": perplex_to_extractor,
    }

    print("Plotting")
    plot_highlight(
        to_highlight={},

        title=title_bland,
        save_path=params["save_path"],
        save_name=params["save_basename"].format('bland'),

        **params_data,
        **params["params_plot"],
    )

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


    print("Computing animation")
    for algo, color in params["to_follow"].items():
        tmp_to_follow = {algo:color}

        plot_follow(

            to_highlight=tmp_to_follow,
            
            save_name=params["save_basename"].format("{}_animated").format(algo),
            save_path=params["save_path"],

            base_title=title_animation,
            
            meta_steps={"train": ['0','1','2','3','4']},

            **params_data,
            **params["params_plot"],
            **params["params_anim"],
        )
