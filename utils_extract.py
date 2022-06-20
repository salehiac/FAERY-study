import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError


def read_file(filename, prefix=None):
    """
    Returns content of file as array
    """

    try:
        with np.load(filename, 'rb') as data:
            return np.array(list(data.values())[0])
    except FileNotFoundError:
        if prefix != "test":
            print("Cannot find the file", filename, end='\r')
        return None


def count_solved(arr):
    """
    Returns the number of solved tasks and their adaptation speeds
    """

    solved, list_min = 0, []
    for line in arr.T:
        strip_line = line[np.where(line >= 0)]

        is_solved = len(strip_line) > 0
        solved += int(is_solved)
        if is_solved is True:
            list_min.append(np.min(strip_line))
    
    return solved, list_min


def get_evolution(path, name, start, end, prefix="train"):
    """
    Retrieves information from the evolution table at given path
    """

    # Number of tasks solved per each individual
    list_nb_solved_ind = []
    # Number of individuals that solved each task
    list_nb_solved_task = []
    # Proportion of tasks solved
    list_prop_solved = []
    # Average minimum adaptations to solve a task
    list_avg_adapt, list_std_adapt = [], []

    for filename in ["{}/evolution_table_{}_{}.npz".format(path+name,prefix,i) for i in range(start, end+1)]:

        arr = read_file(filename, prefix=prefix)
        if arr is None:
            if prefix == "train":   
                print("Could not read", filename, "stopping here for current graph.")
                break
            continue
            
        list_nb_solved_ind.append([len(l[np.where(l>=0)]) for l in arr])
        list_nb_solved_task.append([len(l[np.where(l>=0)]) for l in arr.T])

        solved, list_min = count_solved(arr)  

        list_prop_solved.append(solved / np.shape(arr)[1])
        list_avg_adapt.append(np.mean(list_min))
        list_std_adapt.append(np.std(list_min))

    list_avg_adapt = np.array(list_avg_adapt)
    list_std_adapt = np.array(list_std_adapt)

    return {
        "tasks solved for each individual":list_nb_solved_ind,
        "individuals that solved task":list_nb_solved_task,
        "proportion solved":list_prop_solved,
        "necessary adaptations (mean/std)":(list_avg_adapt, list_std_adapt),
    }


def get_score(path, name, start, end):
    """
    Retrieves the scores from the meta-scores at given path
    """

    list_raw_scores = []
    list_mean_std_0 = [[], []]
    list_mean_std_1 = [[], []]
    list_solved_one_task = []
    for filename in ["{}/meta-scores_train_{}.npz".format(path+name,i) for i in range(start, end+1)]:

        arr = read_file(filename)
        if arr is None:
            break
        init_len = len(arr)

        list_raw_scores.append(arr[:])

        arr = [val for val in arr if val[-1] > float('-inf')]
        
        mean_scores = np.mean(arr, axis=0)
        std_scores = np.std(arr, axis=0)

        list_mean_std_0[0].append(mean_scores[0])
        list_mean_std_1[0].append(mean_scores[1])

        list_mean_std_0[1].append(std_scores[0])
        list_mean_std_1[1].append(std_scores[1])

        list_solved_one_task.append(len(arr) / init_len)
    
    list_mean_std_0 = np.array(list_mean_std_0)
    list_mean_std_1 = np.array(list_mean_std_1)

    return {
        "F0":list_mean_std_0,
        "F1":list_mean_std_1,
        "individuals that solved at least one task":list_solved_one_task,
        "raw scores":list_raw_scores,
    }
    

def save_lone_graph(path, basename, start, end,
                    inner_algo, removed_obj,
                    colors_adapt, colors_scores, colors_solved,
                    save_basename, title, to_path
                    ):
    """
    Saves the graphs for a single algorithm,
    Returns the computed data
    """

    results_obj = []

    for data_name in ["{}_{}".format(inner_algo, removed) for removed in removed_obj]:
        results_obj.append({})

        name = basename.format(data_name)

        data = get_evolution(path=path,
                             name=name,
                             start=start,
                             end=end)

        data_score = get_score(path=path,
                               name=name,
                               start=start,
                               end=end)
        
        list_raw_scores = data_score["raw scores"]
        list_nb_solved = data["tasks solved for each individual"]
        len_learned = len(list_nb_solved)

        solution_per_task = []
        solution_per_task_mean_std = [[],[]]
        for step in range(len(list_raw_scores)):
            scores = list_raw_scores[step]
            solved = list_nb_solved[step]

            new_scores = [
                scores[ind][0]/solved[ind]
                for ind in range(len(scores))
                if solved[ind] != 0
            ]

            solution_per_task.append(new_scores)
            solution_per_task_mean_std[0].append(np.mean(new_scores))
            solution_per_task_mean_std[1].append(np.std(new_scores))
        
        solution_per_task_mean_std[0] = np.array(solution_per_task_mean_std[0])
        solution_per_task_mean_std[1] = np.array(solution_per_task_mean_std[1])

        x_values = range(start, end+1)[:len_learned]

        fig, axs = plt.subplots(figsize=(24,18), nrows=2, ncols=2)
        

        graph_filled(
            ax=axs[0][0],
            toplot_x=x_values,
            list_toplot_y_main=[data["necessary adaptations (mean/std)"][0]],
            list_toplot_y_area=[data["necessary adaptations (mean/std)"][1]],
            list_colors_couple=[colors_adapt],
            
            extr_y=(0, float('inf')),

            xlabel="Generation",
            ylabel="Necessary adaptations\n(meand and std)",       
        )

        graph_filled(
            ax=axs[0][1],
            toplot_x=x_values,
            list_toplot_y_main=[data_score["F0"][0]],
            list_toplot_y_area=[data_score["F0"][1]],
            list_colors_couple=[colors_scores[0]],

            extr_y=(0, float('inf')),

            xlabel="Generation",
            ylabel="F0",
        )

        axs[0][1].tick_params(axis='y', colors=colors_scores[0][0])
        axs[0][1].yaxis.label.set_color(colors_scores[0][0])
        
        ax1_twinx = axs[0][1].twinx()
        graph_filled(   
            ax=ax1_twinx,
            toplot_x=x_values,
            list_toplot_y_main=[data_score["F1"][0]],
            list_toplot_y_area=[data_score["F1"][1]],
            list_colors_couple=[colors_scores[1]],

            extr_y=(float('-inf'), 0),
            
            xlabel="Generation",
            ylabel="F1",       
        )    

        ax1_twinx.tick_params(axis='y', colors=colors_scores[1][0])
        ax1_twinx.yaxis.label.set_color(colors_scores[1][0])

        axs1_0_twinx = axs[1][0].twinx()
        for i, x in enumerate(x_values):
            nb_solved = list_nb_solved[i]
            axs[1][0].scatter([x for _ in range(len(nb_solved))], nb_solved, color=colors_solved[0])
            
        axs1_0_twinx.plot(x_values, data_score["individuals that solved at least one task"], color=colors_solved[1])

        axs[1][0].set_xlabel("Generation")
        axs[1][0].set_ylabel("Number of tasks solved per individual")
        axs1_0_twinx.set_ylabel("% of individual that solved at least a task")

        axs1_0_twinx.set_ylim(0,1)
        axs1_0_twinx.tick_params(axis='y', colors=colors_solved[1])
        axs1_0_twinx.yaxis.label.set_color(colors_solved[1])
        
        axs[1][0].grid(True)
        axs1_0_twinx.grid(True)


        axs[1][1].boxplot(solution_per_task)
        axs[1][1].set_xlabel("Generation")
        axs[1][1].set_ylabel("Average number of solutions per solved tasks\n(specialization)")
        axs[1][1].grid(True)


        plt.suptitle(title.format(data_name))
        plt.savefig("{}/{}.png".format(to_path, save_basename.format(data_name)))

        results_obj[-1]["data"] = data
        results_obj[-1]["data score"] = data_score
        results_obj[-1]["new score (raw, mean_std)"] = [solution_per_task, solution_per_task_mean_std]

    return results_obj


def save_compare_graph(start, end,
                       inner_algo, removed_obj,
                       colors_compare,
                       save_basename_compare, title_compare,
                       results_obj, to_path,
                       ):
    """
    Saves the comparison graph for the ablation
    Returns the computed data
    """

    fig, axs = plt.subplots(figsize=(24,18), nrows=2, ncols=2)
    x_values = range(start, end+1)
    
    common_args = {
        "list_colors_couple":[*colors_compare],
        "list_labels":removed_obj,
        
        "area_alpha":.3,

        "xlabel":"Generation",

        "toplot_x":x_values,
        "show_area":True
    }

    graph_filled(
        ax=axs[0][0],

        list_toplot_y_main=[results_obj[k]["data"]["necessary adaptations (mean/std)"][0] for k in range(len(removed_obj))],
        list_toplot_y_area=[results_obj[k]["data"]["necessary adaptations (mean/std)"][1] for k in range(len(removed_obj))],

        ylabel="Necessary adaptations\n(meand and std)",
        extr_y=(0, float('inf')),

        **common_args
    )

    for k in range(len(removed_obj)):
        raw_scores = results_obj[k]["new score (raw, mean_std)"][0]

        for i, x in enumerate(x_values[:len(raw_scores)]):
            axs[0][1].scatter([x for _ in range(len(raw_scores[i]))], raw_scores[i],
            color=colors_compare[k][0], alpha=.5)

        axs[0][1].plot([], color=colors_compare[k][0], label=str(removed_obj[k]))

    axs[0][1].set_xlabel("Generation")
    axs[0][1].set_ylabel("Average number of solutions per solved tasks\n(specialization)")
    axs[0][1].grid(True)
    axs[0][1].legend()

    graph_filled(
        ax=axs[1][0],

        list_toplot_y_main=[results_obj[k]["data score"]["F0"][0] for k in range(len(removed_obj))],
        list_toplot_y_area=[results_obj[k]["data score"]["F0"][1] for k in range(len(removed_obj))],

        ylabel="F0",
        extr_y=(0, float('inf')),

        **common_args
    )

    graph_filled(
        ax=axs[1][1],

        list_toplot_y_main=[results_obj[k]["data score"]["F1"][0] for k in range(len(removed_obj))],
        list_toplot_y_area=[results_obj[k]["data score"]["F1"][1] for k in range(len(removed_obj))],

        ylabel="F1",
        extr_y=(float('-inf'), 0),

        **common_args
    )

    plt.suptitle(title_compare.format(inner_algo))
    plt.savefig("{}/{}.png".format(to_path, save_basename_compare.format(inner_algo)))

    return {}


def save_animation(inner_algo, colors_compare, end, results_obj, removed_obj, interval, to_path, score_lim=(10,-80)):
    """
    Saves an animation of the individual's scores
    """
    results_obj = results_obj
    removed_obj = removed_obj

    fig = plt.figure(figsize=(20,20), dpi=160)

    def animate(i):
        print("Step {}/{}".format(i, end), end="\r")
        plt.clf()

        RB = [plt.plot([], [], '+')[0] for obj in removed_obj]
        for k, obj in enumerate(removed_obj):
            data_score = results_obj[k]["data score"]["raw scores"]
            scores = np.array([val for val in data_score[min(i, len(data_score)-1)] if val[-1] > float('-inf')])

            try:
                hull = ConvexHull(scores)
                for simplex in hull.simplices:
                    plt.plot(scores[simplex, 0], scores[simplex, 1], color=colors_compare[k][0])
                
                #plt.fill_between(scores[hull.vertices, 0], scores[hull.vertices, 1], color=colors_compare[k][1], alpha=.5)
            except QhullError:
                pass

            x = [val[0] for val in scores]
            y = [val[1] for val in scores]
            
            RB[k].set_data(x, y)
            RB[k].set_label(str(obj))
            RB[k].set_color(colors_compare[k][0])

        plt.legend()
        plt.axis([0,score_lim[0],score_lim[1],1])
        plt.grid(True)
        plt.xlabel("F0")
        plt.ylabel("F1")
        plt.title("Animation of the individuals' scores\nStep {}/{}".format(i, len(data_score)))

        return RB

    my_anim = animation.FuncAnimation(fig, animate, frames=end, interval=interval)
    
    my_anim.save("{}/{}/animated_scores_{}.mp4".format(to_path, inner_algo, inner_algo), writer="ffmpeg")


def graph_filled(ax,

                toplot_x, list_toplot_y_main, list_toplot_y_area,
                list_colors_couple=[("indianred", "pink")],
                list_labels=None,

                xlabel: str = "", ylabel: str = "",
                extr_y=(float('-inf'), float('inf')),
                extr_margin=(0.8,1.2),
                area_alpha=.8,

                show_area=True,

                ) -> None:

    """
    Easily define graphs with filled area
    """
    
    for i in range(len(list_toplot_y_main)):

        color_couple = list_colors_couple[i%len(list_colors_couple)]
        label = "" if (list_labels is None) or (i > len(list_labels)) else str(list_labels[i])

        ax.plot(toplot_x[:len(list_toplot_y_main[i])], list_toplot_y_main[i],
                color=color_couple[0],
                label=label
        )
        
        if show_area is True:
            ax.fill_between(toplot_x[:len(list_toplot_y_main[i])],
                            [min(extr_y[1], val) for val in list_toplot_y_main[i] + list_toplot_y_area[i]],
                            [max(extr_y[0], val) for val in list_toplot_y_main[i] - list_toplot_y_area[i]],
                            color=color_couple[1],
                            alpha=area_alpha
            )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_ylim(bottom=None if extr_y[0] == float('-inf') else extr_margin[0] * extr_y[0],
                top=None if extr_y[1] == float('inf') else extr_margin[1] * extr_y[1])

    ax.grid(True)

    if list_labels is not None:
        ax.legend()