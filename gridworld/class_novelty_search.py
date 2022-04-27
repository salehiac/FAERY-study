from class_inner_algorithm import InnerAlgorithm

class NoveltySearch(InnerAlgorithm):
    """
    Simple novelty search class
    """

    def __init__(
        self, environment, nb_generations,
        population_size, offspring_size,
        meta_generation,
        ):

        super().__init__(
            environment, nb_generations,
            population_size, offspring_size,
            meta_generation,
        )

        