import numpy as np
import os
import matplotlib.pyplot as plt


path = "../data backup/"
files = ["NS_one", "RAND_one"]
title = "Comparing random and NS on assembly-v2"
colors = ["dodgerblue", "indianred"]

save_basename = "Results_compare"

fig, axs = plt.subplots(2, figsize=(16,12))

for i, f in enumerate(files):

    try:
        with np.load(path+f+"/evolution_table_train_0.npz", 'rb') as data:
            arr = np.array(list(data.values())[0])
    except FileNotFoundError:
        print("Not Found")
        continue

    solved, list_min = 0, []
    for line in arr.T:
        strip_line = line[np.where(line >= 0)]

        is_solved = len(strip_line) > 0
        solved += int(is_solved)
        if is_solved is True:
            list_min.append(np.min(strip_line))

    prop_solved = solved / np.shape(arr)[1]
    avg_adapt = np.mean(list_min)
    std_adapt = np.std(list_min)


    axs[0].bar(x=i, height=prop_solved, width=.75, color=colors[i], label=f)
    axs[1].bar(x=i, height=avg_adapt, yerr=std_adapt, width=.75, color=colors[i], label=f)


axs[0].set_ylabel("% of solved environments")
axs[0].grid(True)
axs[0].legend()


axs[1].set_ylabel("Necessary adaptations\n(mean and std)")
axs[1].grid(True)
axs[1].legend()

plt.suptitle(title)

plt.savefig("../data backup/Images/{}.png".format(save_basename))
