import numpy as np
import matplotlib.pyplot as plt

from utils_extract import *


algo_types = ["NS"]
removed_obj = [-1,0,1]

start, end = 0, 99
path, basename = "../data backup/FAERY/", "FAERY_{}"

colors_adapt = ("dodgerblue", "lightblue")
colors_scores = [("green","yellowgreen"), ("red","darksalmon")]
colors_solved = ("gray", "dodgerblue")
colors_new_score = ("dodgerblue", "lightblue")

colors_compare = [("dodgerblue", "lightblue"), ("red","darksalmon"), ("green", "yellowgreen")]

title_solo = "Performances de FAERY appliqué à {} sur 100x200 steps\nmoyenne sur 25 tâches"
title_compare = "Comparison {}"

save_basename_solo = "Results_{}"
save_basename_compare = "Results_compare_{}"


for inner_algo in algo_types:

    results_obj = save_lone_graph(
        path=path,
        basename=basename,
        start=start, end=end,
        inner_algo=inner_algo,
        removed_obj=removed_obj,
        colors_adapt=colors_adapt,
        colors_scores=colors_scores,
        colors_solved=colors_solved,
        save_basename=save_basename_solo,
        title=title_solo
    )

    fig, axs = plt.subplots(figsize=(24,18), nrows=2, ncols=2)
    x_values = range(start, end+1)

    graph_filled(
        ax=axs[0][0],

        toplot_x=x_values,
        list_toplot_y_main=[results_obj[k]["data"]["necessary adaptations (mean/std)"][0] for k in range(len(removed_obj))],
        list_toplot_y_area=[results_obj[k]["data"]["necessary adaptations (mean/std)"][1] for k in range(len(removed_obj))],
        list_colors_couple=[*colors_compare],
        list_labels=removed_obj,
        
        extr_y=(0, float('inf')),
        area_alpha=.5,

        xlabel="Generation",
        ylabel="Necessary adaptations\n(meand and std)",

    )

    for k in range(len(removed_obj)):
        raw_scores = results_obj[k]["new score (raw, mean_std)"][0]

        for i, x in enumerate(x_values):
            axs[0][1].scatter([x for _ in range(len(raw_scores[i]))], raw_scores[i],
            color=colors_compare[k][0], alpha=.5)

        axs[0][1].plot([], color=colors_compare[k][0], label=str(removed_obj[k]))

    axs[0][1].set_xlabel("Generation")
    axs[0][1].set_ylabel("Average number of solutions per solved tasks\n(specialization)")
    axs[0][1].grid(True)
    axs[0][1].legend()

    graph_filled(
        ax=axs[1][0],

        toplot_x=x_values,
        list_toplot_y_main=[results_obj[k]["data score"]["F0"][0] for k in range(len(removed_obj))],
        list_toplot_y_area=[results_obj[k]["data score"]["F0"][1] for k in range(len(removed_obj))],
        list_colors_couple=[*colors_compare],
        list_labels=removed_obj,
        
        extr_y=(0, float('inf')),
        area_alpha=.5,

        xlabel="Generation",
        ylabel="F0",

    )

    graph_filled(
        ax=axs[1][1],

        toplot_x=x_values,
        list_toplot_y_main=[results_obj[k]["data score"]["F1"][0] for k in range(len(removed_obj))],
        list_toplot_y_area=[results_obj[k]["data score"]["F1"][1] for k in range(len(removed_obj))],
        list_colors_couple=[*colors_compare],
        list_labels=removed_obj,
        
        extr_y=(float('-inf'), 0),
        area_alpha=.5,

        xlabel="Generation",
        ylabel="F1",

    )

    plt.suptitle(title_compare.format(inner_algo))
    plt.savefig("../data backup/Images/{}.png".format(save_basename_compare.format(inner_algo)))
