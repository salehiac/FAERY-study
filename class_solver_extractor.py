from utils_tsne import get_files, order_str_int


class SolverExtractor:
    """
    Able to flatten an extracted solvers_dict and to unpack it.
    Useful for plotting purposes
    """

    def __init__(self, solvers_dict=None, load_path=None, max_samples=5000) -> None:
        self.solvers_dict = solvers_dict if solvers_dict is not None \
            else self._get_files(load_path, max_samples=max_samples) if load_path is not None \
                else None

        if self.solvers_dict is None:
            raise ValueError("Please provide a path or an already loaded solvers_dict.")

        
        # Hard coded fix....
        try:
            del self.solvers_dict["FAERY_buttonpress"]
        except:
            pass
        ## Saved in ordred format, unlike solvers_dict
        self.algorithms = list(self.solvers_dict.keys())
        self.type_runs = ('train', 'test')
        self.meta_steps = {
            type_run: order_str_int(self.solvers_dict[self.algorithms[-1]][type_run].keys())
            for type_run in self.type_runs
        }

        ## Positions of the populations (start, end) in flattened list
        self.position_in_list = {}

        self.flattened = self._flatten()
        self.list = self.flattened
    
    def _get_files(self, path, max_samples) -> dict:
        return get_files(path, basename="solvers", max_samples=max_samples)[0]
    
    def _flatten(self) -> list:
        """
        Flattens the input solvers_dict
        """

        solvers_list = []

        for algorithm in self.algorithms:
            for type_run in self.type_runs:
                for meta_step in self.meta_steps[type_run]:
                    inter_dict = self.solvers_dict[algorithm][type_run][meta_step]
                    for inner_step in order_str_int(inter_dict.keys()):
                        self.position_in_list[(
                            algorithm,
                            type_run,
                            meta_step,
                            inner_step
                        )] = (
                            len(solvers_list), #start
                            len(solvers_list) + len(inter_dict[inner_step]) #end
                        )

                        solvers_list += list(inter_dict[inner_step])
        
        return solvers_list

    def unpack(self, input_list) -> dict:
        """
        Unpacks a given list according the saved solvers_dict
        """

        output_dict = {}

        for algorithm in self.algorithms:
            output_dict[algorithm] = {}
            for type_run in self.type_runs:
                output_dict[algorithm][type_run] = {}
                for meta_step in self.meta_steps[type_run]:
                    output_dict[algorithm][type_run][meta_step] = {}
                    for inner_step in order_str_int(self.solvers_dict[algorithm][type_run][meta_step].keys()):
                        start, end = self.position_in_list[
                            algorithm, type_run, meta_step, inner_step
                        ]

                        output_dict[algorithm][type_run][meta_step][inner_step] = \
                            input_list[start:end]

        return output_dict

    def __check_valid(self, goal, current, contained=False):
        """
        Checks if the goal corresponds to the current query
        """

        return (goal is None) or (goal in current if contained is True else goal == current)

    def get_params(self, input_algorithm=None, input_type=None, input_meta=None, input_step=None):
        """
        Returns all the solvers from the given parameters
        """

        solvers_from_algo = []
        for (algorithm, type_run, meta_step, inner_step), (start, end) in self.position_in_list.items():
            if all([
                self.__check_valid(input_algorithm, algorithm, True),
                self.__check_valid(input_type, type_run),
                self.__check_valid(input_meta, meta_step),
                self.__check_valid(input_step, inner_step)
                ]) is True: solvers_from_algo += self.list[start:end]

        return solvers_from_algo

    def find_algorithm(self, abreviation):
        """
        Returns the full name of the given abreviation
        """

        for algorithm in self.algorithms:
            if abreviation in algorithm:
                return algorithm
        
        raise ValueError("Algorithm {} not found amongst {}".format(abreviation, self.algorithms))


if __name__=="__main__":

    s = SolverExtractor(load_path="./data")

    flattened = s.list
    unpacked = s.unpack(flattened)

    for algorithm in s.algorithms:
        for type_run in s.type_runs:
            for meta_step in s.meta_steps[type_run]:
                inter_dict = s.solvers_dict[algorithm][type_run][meta_step]
                for inner_step in order_str_int(inter_dict.keys()):
                    so = s.solvers_dict[algorithm][type_run][meta_step][inner_step]
                    uo = unpacked[algorithm][type_run][meta_step][inner_step]

                    assert so == uo, "Unpacking failed"

    assert sum([len(s.get_params(algo)) for algo in s.algorithms]) == len(s.list), \
        "Recovery of algorithms failed"
