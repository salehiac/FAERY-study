import json
import matplotlib.pyplot as plt

from utils_misc import get_path
from utils_extract import *


with open(get_path(default="results.json"), 'r') as f:
    params = json.load(f)


for k_fig in params["filerange"]:
    
    name = params["basename"].format(k_fig)

    fig, axs = plt.subplots(figsize=(16,12), nrows=2)
    
    for prefix in ("test", "train"):

        data = get_evolution(path=params["path"],
                             name=name,
                             start=params["start"],
                             end=params["end"],
                             prefix=prefix)
        
        x_prop = range(params["start"], params["end"]+1) if prefix == "train" else range(params["start"], params["end"]+1, 10)

        axs[0].plot(x_prop, data["proportion solved"],
                    color=params["colors"][prefix][0], label=prefix)
        
        graph_filled(ax=axs[1],
                     toplot_x=x_prop,
                     list_toplot_y_main=[data["necessary adaptations (mean/std)"][0]],
                     list_toplot_y_area=[data["necessary adaptations (mean/std)"][1]],
                     colors_couple=[params["colors"][prefix]],
                     list_labels=[prefix],
                     xlabel="Generation",
                     ylabel="Necessary adaptations\n(mean and std)",
                    )

        axs[0].set_xlabel("Generation")
        axs[0].set_ylabel("% of solved environments")
        axs[0].set_ylim(0, 1.2)
        axs[0].grid(True)
        axs[0].legparams["end"]()


    plt.supparams["title"](params["title"].format(k_fig))

    plt.savefig("../data backup/Images/{}.png".format(params["save_basename"].format(k_fig)))
