import json
import argparse

from utils_extract import *

def get_path(parser=None, to_parse=True, default="./results.json", verbose=True):
    """
    Returns the queried path
    """

    if parser is None:
        parser = parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_params",
        type=str,
        help="path to the json file",
        default=default
    )

    path = parser.parse_args().path_params
    if path[-5:] != ".json":    path += ".json"

    if verbose is True:
        print("Loaded parameters from {}".format(path))

    return path if to_parse is True else parser


with open(get_path(default="results_ablation.json"), 'r') as f:
    params = json.load(f)


for inner_algo in params["start_end"].keys():

    save_basepath = "{}/".format(inner_algo)
    start, end = params["start_end"][inner_algo]
    
    if params["solo"] is True:
        print("Saving lone graph..", end='\r')
        results_obj = save_lone_graph(
            path=params["path"],
            basename=params["basename"],
            start=start, end=end,
            inner_algo=inner_algo,
            removed_obj=params["removed_obj"],
            colors_adapt=params["colors_adapt"],
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
            removed_obj=params["removed_obj"],
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
            removed_obj=params["removed_obj"],
            interval=500,
            to_path=params["save_path"],
        )
    print(40 * " ", end='\r')
    print("Done")
