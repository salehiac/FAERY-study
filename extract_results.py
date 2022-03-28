import numpy as np
import matplotlib.pyplot as plt

from utils_extract import *


filerange = ["QD"]
start, end = 0, 61
path, basename = "data/FAERY/", "FAERY_{}_-1"
title = "Performances de FAERY appliqué à {} sur 75x200 steps\nmoyenne sur 25 tâches"
colors = {"train": ("dodgerblue", "lightblue"), "test": ("indianred", "pink")}

save_basename = "Results_{}"

for k_fig in filerange:
    
    name = basename.format(k_fig)

    fig, axs = plt.subplots(figsize=(16,12), nrows=2)
    
    for prefix in ("test", "train"):

        data = get_evolution(path=path,
                             name=name,
                             start=start,
                             end=end,
                             prefix=prefix)
        
        x_prop = range(start, end+1) if prefix == "train" else range(start, end+1, 10)

        axs[0].plot(x_prop, data["proportion solved"],
                    color=colors[prefix][0], label=prefix)
        
        graph_filled(ax=axs[1],
                     toplot_x=x_prop,
                     list_toplot_y_main=[data["necessary adaptations (mean/std)"][0]],
                     list_toplot_y_area=[data["necessary adaptations (mean/std)"][1]],
                     list_colors_couple=[colors[prefix]],
                     list_labels=[prefix],
                     xlabel="Generation",
                     ylabel="Necessary adaptations\n(mean and std)",
                    )

        axs[0].set_xlabel("Generation")
        axs[0].set_ylabel("% of solved environments")
        axs[0].set_ylim(0, 1.2)
        axs[0].grid(True)
        axs[0].legend()


    plt.suptitle(title.format(k_fig))

    plt.savefig("../data backup/Images/{}.png".format(save_basename.format(k_fig)))
