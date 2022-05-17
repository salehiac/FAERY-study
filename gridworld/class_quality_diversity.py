from scipy.spatial import KDTree

from class_toolbox_algorithm import ToolboxAlgorithmGridWorld
from class_novelty_archive import NoveltyArchive


class QualityDiversity(ToolboxAlgorithmGridWorld):
    """
    Simple QD class
    """

    def __init__(
        self,
        
        *args,

        archive = {
            "type":NoveltyArchive,
            "parameters":{
                "neighbouring_size":15,
                "max_size":None
            }
        },

        selection_weights=(1,1),

        **kwargs
        ):
        """
        archive : type and parameters of implemented archive

        """

        super().__init__(
            *args,
            selection_weights=selection_weights,
            **kwargs
        )

        self.archive = archive["type"](**archive["parameters"])
        self.env_kd_tree = KDTree(self.environment.reward_coords)
    
    def _update_fitness(self, population):
        """
        Evaluates a population of individuals and saves their novelty and quality as fitness
        """

        fitnesses = self.toolbox.map(self._evaluate, population)
        
        # Updating the archive
        for ag in population:
            next(fitnesses)
            self.archive.update(ag)

        # Computing the population's novelty
        if self.archive.get_size() >= self.archive.neigh_size:
            for ag in population:
                ag.fitness.values = (
                    self.archive.get_novelty(ag.behavior)[0],
                    self.env_kd_tree.query(ag.behavior, k=1)[0]
                )
            return True
        
        for ag in [ind for ind in population if ind.fitness.valid is False]:
            ag.fitness.values = (
                -1 * self.selection_weights[0] * float("inf"),
                -1 * self.selection_weights[1] * float("inf")
            )

        return False
    
    def reset(self):
        """
        Resets the algorithm and the archive
        """

        super().reset()
        self.archive.reset()
        

if __name__ == "__main__":

    from environment.class_gridagent import GridAgentNN, GridAgentGuesser

    compute_NN = False
    compute_Guesser = True

    # For NN agents
    if compute_NN is True:

        import torch
        from deap import tools

        ns = QualityDiversity(
            nb_generations=20,
            population_size=10,
            offspring_size=10,

            archive={
                "type":NoveltyArchive,
                "parameters":{
                    "neighbouring_size":2,
                    "max_size":None
                }
            },

            ag_type=GridAgentNN,

            creator_parameters={
                "individual_name":"NNIndividual",
                "fitness_name":"NNFitness",
             },
        )

    # For guesser agents (those in the article)
    if compute_Guesser is True:
      
        ns = QualityDiversity(
            nb_generations=100,
            population_size=20,
            offspring_size=20,

            ag_type=GridAgentGuesser,
        )

    pop, log, hof = ns(show_history=True)
    print(log)

    ns.environment.visualise_as_grid(
        list_state_hist=[[behavior] for behavior in ns.archive.behaviors],
        show_traj=True,
        show_start=False,
        show_end=True,
        show=True
    )
