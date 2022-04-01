import json

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
    plot_highlight(

        to_highlight=params["to_highlight"],

        title=title_highlight,
        save_path=params["save_path"],
        save_name=params["save_basename"].format("highlighted"),

        **params_data,
        **params["params_plot"],
    )

    print("Computing animation")
    for algo, color in params["to_follow"].items():
        temp_to_follow = {algo:color}

        plot_follow(

            to_highlight=temp_to_follow,
            meta_steps={"train": extractor.meta_steps["train"]},

            base_title=title_animation,
            save_path=params["save_path"],
            save_name=params["save_basename"].format("{}_animated".format(algo)),

            **params_data,
            **params["params_plot"],
            **params["params_anim"],
        )
