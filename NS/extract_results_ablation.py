import numpy as np
import matplotlib.pyplot as plt

filerange = [-1, 0, 1]
start, end = 0, 44
path, basename = "../data backup/FAERY/", "FAERY_QD_{}"
title = ""
colors = ("dodgerblue", "lightblue")
colors_scores = {1:("chartreuse","yellowgreen"), 2:("indianred","darksalmon")}

save_basename = "Results_{}"

for k_fig in filerange:
    
    name = basename.format(k_fig)

    fig, axs = plt.subplots(figsize=(16,18), nrows=3)

    
    list_prop_solved, list_avg_adapt, list_std_adapt = [], [], []
    for filename in [path+name+"/evolution_table_train_{}.npz".format(i) for i in range(start, end+1)]:

        try:
            with np.load(filename, 'rb') as data:
                arr = np.array(list(data.values())[0])
        except FileNotFoundError:
            raise FileNotFoundError("Cannot find the file:", filename)

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

    list_mean_std_1 = [[],[]]
    list_mean_std_2 = [[],[]]
    for filename in [path+name+"/meta-scores_train_{}.npz".format(i) for i in range(start, end+1)]:

        try:
            with np.load(filename, 'rb') as data:
                arr = np.array(list(data.values())[0])
        except FileNotFoundError:
            raise FileNotFoundError("Cannot find the file:", filename)

        mean_scores = np.mean(arr, axis=0)
        std_scores = np.std(arr, axis=0)

        list_mean_std_1[0].append(mean_scores[0])
        list_mean_std_2[0].append(mean_scores[1])

        list_mean_std_1[1].append(std_scores[0])
        list_mean_std_2[1].append(std_scores[1])
    
    list_mean_std_1 = np.array(list_mean_std_1)
    list_mean_std_2 = np.array(list_mean_std_2)
    print(list_mean_std_2)
    x_prop = range(len(list_prop_solved))
    axs[0].plot(x_prop, list_prop_solved,
                color=colors[0])
    axs[1].plot(x_prop, list_avg_adapt, color=colors[0])
    axs[1].fill_between(x_prop, list_avg_adapt + list_std_adapt,
                        [max(val, 0) for val in list_avg_adapt - list_std_adapt], color=colors[1])
    
    axs2_2 = axs[2].twinx()
    axs[2].plot(x_prop, list_mean_std_1[0], color=colors_scores[1][0])
    axs[2].fill_between(x_prop, list_mean_std_1[0] + list_mean_std_1[1],
                        [max(val, 0) for val in list_mean_std_1[0] - list_mean_std_1[1]], color=colors_scores[1][1])
    
    axs2_2.plot(x_prop, list_mean_std_2[0], color=colors_scores[2][0])
    axs2_2.fill_between(x_prop, list_mean_std_2[0] + list_mean_std_2[1],
                        [max(val, 0) for val in list_mean_std_2[0] - list_mean_std_2[1]], color=colors_scores[2][1])

    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("% of solved environments")
    axs[0].set_ylim(0, 1.2)
    axs[0].grid(True)
    axs[0].legend()

    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Necessary adaptations\n(mean and std)")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].set_xlabel("Generation")
    axs[2].set_ylabel("F0")
    axs2_2.set_ylabel("F1")
    axs[2].grid(True)
    axs[2].legend()

    plt.suptitle(title.format(k_fig))

    plt.savefig("../data backup/Images/{}.png".format(save_basename.format(k_fig)))
