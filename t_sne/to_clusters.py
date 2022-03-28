import os
import sys
import pickle
import functools
import numpy as np

from numpy.random import choice
from sklearn.manifold import TSNE

from t_sne.utils_tsne import *


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: this_script <ns_logs_dir> <meta_pop_dir>  where ns_logs_dir is a directory containing a (large) number of NS_log_<pid> directories, and meta-pop_dir is the meta population to project")

    path = sys.argv[1]

    if len(sys.argv) == 3:  # assume that we also have a prior population that we also want to project
        with open(sys.argv[2], "rb") as pfl:
            prior_pop = pickle.load(pfl)
    else:
        prior_pop = None

    fns = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(path)) for f in fn]

    fns = [x for x in fns if "population_gen_" in x]

    gen_to_pop = {}

    for pop_fn in fns:
        dig = [x for x in pop_fn[pop_fn.find(
            "population_gen"):] if x.isdigit()]
        dig = int(functools.reduce(lambda x, y: x+y, dig, ""))
        if dig not in gen_to_pop:
            gen_to_pop[dig] = [pop_fn]
        else:
            gen_to_pop[dig].append(pop_fn)

    keys = list(gen_to_pop.keys())
    # that probably does not really makes sense
    inversly_propotional_to_freq = False
    if inversly_propotional_to_freq:
        weights = [max(1/len(gen_to_pop[i]), 0.1) for i in keys]
        s_w = sum(weights)
        normalised_weights = [x/s_w for x in weights]
    else:
        normalised_weights = [1/len(keys) for x in keys]
    num_samples = 5000  # note that we might have more or less than this later as 1) there might be multiple solvers per sample 2) we remove duplicates using set later on
    print(f"sampling {num_samples} popluations")
    draw = choice(keys, num_samples, p=normalised_weights)

    selected = []

    for k in draw:
        sel = np.random.choice(gen_to_pop[k])
        selected.append(sel)

    print("getting solvers")
    solvers = get_solvers(selected)

    solvers = list(set(solvers))

    solver_weights = [
        np.array(x.get_flattened_weights()).reshape(1, -1) for x in solvers]
    solver_weights = np.concatenate(solver_weights, 0)

    if prior_pop is not None:
        prior_pop_weights = [
            np.array(x.get_flattened_weights()).reshape(1, -1) for x in prior_pop]
        prior_pop_weights = np.concatenate(prior_pop_weights, 0)

        weights = np.concatenate([solver_weights, prior_pop_weights], 0)
    else:
        weights = solver_weights

    #pca = PCA(n_components=50)
    # weight_pcs=pca.fit_transform(weights)
    # pdb.set_trace()

    print("creating embedding")
    emb = TSNE(n_components=2).fit_transform(weights)
    #emb= TSNE(n_components=2).fit_transform(weight_pcs)
    logdir = "/tmp/embedding_dir_local/"
    np.savez_compressed(logdir+"/embedding", emb)
    with open(logdir+"/embedding_info", "w") as fl:
        fl.write(f"num_appended_prior_weights=={prior_pop_weights.shape[0]}")
