import json

from utils_extract import *


with open(get_path(default="params_performance.json"), 'r') as f:
    params = json.load(f)


for inner_algo in params["start_end"].keys():

    save_basepath = "{}/".format(inner_algo)
    create_path("{}/{}".format(params["save_path"], save_basepath))
    
    start, end = params["start_end"][inner_algo]
    
    print("Saving lone graph..", end='\r')
    results_obj = save_lone_graph(
        path=params["path"],
        basename=params["basename"],
        start=start, end=end,
        inner_algo=inner_algo,
        suffix=params["suffix"],
        colors_adapt=params["colors_adapt"],
        colors_adapt_test=params["colors_adapt_test"],
        colors_scores=params["colors_scores"],
        colors_solved=params["colors_solved"],
        save_basename=save_basepath + params["save_basename_solo"],
        title=params["title_solo"],
        to_path=params["save_path"],
    )

    if params["compare"] is True:
        print("Saving compare graph..", end='\r')
        _ = save_compare_graph(
            start=start, end=end,
            inner_algo=inner_algo,
            suffix=params["suffix"],
            colors_compare=params["colors_compare"],
            save_basename_compare=save_basepath + params["save_basename_compare"],
            title_compare=params["title_compare"],
            results_obj=results_obj,
            to_path=params["save_path"],
        )

    if params["animate"] is True:
        print("Saving the scores' animation..", end='\r')
        save_animation(
            inner_algo=inner_algo,
            colors_compare=params["colors_compare"],
            end=end,
            results_obj=results_obj,
            suffix=params["suffix"],
            interval=500,
            to_path=params["save_path"],
            score_lim=params["score_lim"],
        )
    print("\nDone")
