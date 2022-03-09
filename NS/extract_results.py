import numpy as np
import matplotlib.pyplot as plt

start, end = 0, 49
path, name = "../data backup/", "meta-learning_ns_3"
title = "Performances de NS simple sur 50x150 steps\nmoyenne sur 10 tâches"
colors = {"train": ("dodgerblue", "lightblue"), "test": ("indianred", "pink")}

fig, axs = plt.subplots(nrows=2)

for prefix in ("test", "train"):
    list_prop_solved, list_avg_adapt, list_std_adapt = [], [], []
    for filename in [path+name+"/evolution_table_{}_{}.npz".format(prefix, i) for i in range(start, end+1)]:

        try:
            with np.load(filename, 'rb') as data:
                arr = np.array(list(data.values())[0])
        except FileNotFoundError:
            continue

        solved, list_min = 0, []
        for line in arr.T:
            strip_line = line[np.where(line >= 0)]

            is_solved = len(strip_line) > 0
            solved += int(is_solved)
            if is_solved is True:
                list_min.append(np.min(strip_line))

        list_prop_solved.append(solved / np.shape(arr)[1])
        list_avg_adapt.append(np.mean(list_min))
        list_std_adapt.append(np.std(list_min))

    list_avg_adapt = np.array(list_avg_adapt)
    list_std_adapt = np.array(list_std_adapt)

    x_prop = range(len(list_prop_solved)) if prefix == "train" else [
        0, 10, 20, 30, 50]
    axs[0].plot(x_prop, list_prop_solved,
                color=colors[prefix][0], label=prefix)
    axs[1].plot(x_prop, list_avg_adapt, color=colors[prefix][0], label=prefix)
    axs[1].fill_between(x_prop, list_avg_adapt + list_std_adapt,
                        list_avg_adapt - list_std_adapt, color=colors[prefix][1])

    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("% of solved environments")
    axs[0].set_ylim(0, 1.2)
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Necessary adaptations\n(mean and std)")
    axs[1].grid(True)
    axs[1].legend()

plt.suptitle(title)
plt.show()
