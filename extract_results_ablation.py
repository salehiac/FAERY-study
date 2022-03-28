from utils_extract import *


start_end = {
    "QD":(0,99),
    "NS":(0,99),
    "RANDOM":(0,99)
    }
removed_obj = [-1, 0, 1]
path, basename = "data/FAERY/", "FAERY_{}"

solo = True
compare = True
animate = True

colors_adapt = ("dodgerblue", "lightblue")
colors_scores = [("green","yellowgreen"), ("red","darksalmon")]
colors_solved = ("gray", "dodgerblue")
colors_new_score = ("dodgerblue", "lightblue")

colors_compare = [("dodgerblue", "lightblue"), ("red","darksalmon"), ("green", "yellowgreen")]

title_solo = "Performances de FAERY appliqué à {} sur 100x200 steps\nmoyenne sur 25 tâches"
title_compare = "Comparison {}"

save_basename_solo = "Results_{}"
save_basename_compare = "Results_compare_{}"


for inner_algo in start_end.keys():

    save_basepath = "{}/".format(inner_algo)
    start, end = start_end[inner_algo]
    
    if solo is True:
        results_obj = save_lone_graph(
            path=path,
            basename=basename,
            start=start, end=end,
            inner_algo=inner_algo,
            removed_obj=removed_obj,
            colors_adapt=colors_adapt,
            colors_scores=colors_scores,
            colors_solved=colors_solved,
            save_basename=save_basepath + save_basename_solo,
            title=title_solo
        )

    if compare is True:
        _ = save_compare_graph(
            start=start, end=end,
            inner_algo=inner_algo,
            removed_obj=removed_obj,
            colors_compare=colors_compare,
            save_basename_compare=save_basepath + save_basename_compare,
            title_compare=title_compare,
            results_obj=results_obj,
        )

    if animate is True:
        save_animation(
            inner_algo=inner_algo,
            colors_compare=colors_compare,
            end=end,
            results_obj=results_obj,
            removed_obj=removed_obj,
            interval=500,
        )
