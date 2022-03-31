import argparse

import matplotlib.pyplot as plt

from t_sne.utils_tsne import *
from t_sne.class_solver_extractor import SolverExtractor


nb_samples = 5000
perplexities = [25, 50, 75, 100]

to_highlight = {
    "QD_-1":"teal",
    #"QD_0":"dodgerblue",
    #"QD_1":"skyblue",
    #"NS_-1":"firebrick",
    #"NS_0":"orangered",
    #"NS_1":"lightcoral",
}

save_path, save_basename = "data/Images/solvers", "TNSE_{}"

title_bland = "TSNE on the behavior descriptors of {} sampled solvers\nin Metworld assembly-v2".format(
    nb_samples)
title_highlight = title_bland

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='TSNE visualization')

    parser.add_argument(
        "--meta_dir",
        type=str,
        help="path to meta-learning directory",
        default="./data/solvers"
    )

    args = parser.parse_args()

    print("Retrieving files in {}".format(args.meta_dir))
    extractor = SolverExtractor(load_path=args.meta_dir)

    print("Computing TSNEs")
    perplex_to_tsne = compute_tsne(
        input_list=extractor.list,
        perplexities=perplexities,
        max_samples=nb_samples,
        verbose=True
    )

    print("Unpacking TSNEs")
    perplex_to_extractor = {
        p:SolverExtractor(solvers_dict = extractor.unpack(perplex_to_tsne[p]))
        for p in perplex_to_tsne.keys()
    }

    print("Plotting")
    plot_highlight(
        perplexities=perplexities,
        perplex_to_extractor=perplex_to_extractor,
        to_highlight={},
        title=title_bland,
        save_path=save_path,
        save_name=save_basename.format("bland"),
    )

    print("Plotting highlight")
    plot_highlight(
        perplexities=perplexities,
        perplex_to_extractor=perplex_to_extractor,
        to_highlight=to_highlight,
        title=title_highlight,
        save_path=save_path,
        save_name=save_basename.format("highlighted"),
    )

    #FOLLOW META
    #FOLLOW INNER
