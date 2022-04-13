import random
import copy

import novelty_search.class_archive as class_archive
import novelty_search.class_novelty_estimators as class_novelty_estimators

from novelty_search.class_novelty_search import NoveltySearch


def _mutate_initial_prior_pop(parents, mutator, agent_factory):
    """
    to avoid a population where all agents have init that is too similar
    """
    parents_as_list = [(x._idx, x.get_flattened_weights()) for x in parents]
    parents_to_mutate = range(len(parents_as_list))
    mutated_genotype = [
        (parents_as_list[i][0], mutator(copy.deepcopy(parents_as_list[i][1])))
        for i in parents_to_mutate
    ]  # deepcopy is because of deap

    num_s = len(parents_as_list)
    mutated_ags = [
        agent_factory(x) for x in range(num_s)
    ]  # we replace the previous parents so we also replace the _idx
    for i in range(num_s):
        mutated_ags[i]._parent_idx = -1
        mutated_ags[i]._root = mutated_ags[
            i]._idx  # it is itself the root of an evolutionnary path
        mutated_ags[i].set_flattened_weights(mutated_genotype[i][1][0])
        mutated_ags[i]._created_at_gen = -1

    return mutated_ags


def _mutate_prior_pop(n_offspring, parents, mutator, agent_factory,
                      total_num_ags):
    """
    mutations for the prior (top-level population)

    total_num_ags  is for book-keeping with _idx, it replaces the previous class variable num_instances for agents which was problematic with multiprocessing/multithreading
    """

    parents_as_list = [(x._idx, x.get_flattened_weights(), x._root)
                       for x in parents]
    parents_to_mutate = random.choices(range(len(parents_as_list)),
                                       k=n_offspring)
    mutated_genotype = [
        (parents_as_list[i][0], mutator(copy.deepcopy(parents_as_list[i][1])),
         parents_as_list[i][2]) for i in parents_to_mutate
    ]  # deepcopy is because of deap

    num_s = n_offspring
    mutated_ags = [agent_factory(total_num_ags + x) for x in range(num_s)]
    kept = random.sample(range(len(mutated_genotype)), k=num_s)
    for i in range(len(kept)):
        mutated_ags[i]._parent_idx = -1  # we don't care
        mutated_ags[i]._root = mutated_ags[
            i]._idx  # it is itself the root of an evolutionnary path.
        # Each individual needs to be its own root, otherwise the evolutionnary path from the inner loops can lead to a meta-individual that
        # has been removed from the population
        mutated_ags[i].set_flattened_weights(mutated_genotype[kept[i]][1][0])
        mutated_ags[i]._created_at_gen = -1  # we don't care

    return mutated_ags


def ns_instance(sampler, population, mutator, inner_selector, make_ag,
                G_inner, top_level_log, prefix_tuple):
    """
    problems are now sampled in the NS constructor
    """
    # those are population sizes for the QD algorithms, which are different from the top-level one
    population_size = len(population)
    offsprings_size = population_size

    nov_estimator = class_novelty_estimators.ArchiveBasedNoveltyEstimator(k=15)
    arch = class_archive.ListArchive(max_size=5000,
                                growth_rate=6,
                                growth_strategy="random",
                                removal_strategy="random")

    ns = NoveltySearch(
        archive=arch,
        nov_estimator=nov_estimator,
        mutator=mutator,
        problem=None,
        selector=inner_selector,
        n_pop=population_size,
        n_offspring=offsprings_size,
        agent_factory=make_ag,
        map_type="scoop",  # or "std"
        compute_parent_child_stats=0,
        initial_pop=[x for x in population],
        problem_sampler=sampler,
        top_level_log=top_level_log,
        prefix_tuple=prefix_tuple)

    # do NS
    nov_estimator.log_dir = ns.log_dir_path
    ns.disable_tqdm = True
    ns.save_archive_to_file = False
    parents, solutions = ns(
        iters=G_inner,
        stop_on_reaching_task=False,  # should not be False in the current implementation)
        save_checkpoints=0
    )  # save_checkpoints is not implemented but other functions already do its job

    if not len(solutions.keys()):  # environment wasn't solved
        return [], -1, parents

    assert len(solutions.keys(
    )) == 1, "solutions should only contain solutions from a single generation"
    depth = list(solutions.keys())[0]

    roots = [sol._root for sol in solutions[depth]]

    return roots, depth, parents