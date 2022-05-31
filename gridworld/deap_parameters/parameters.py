import torch
import numpy as np

from deap import tools


# Would be better if single object, you could pass type as argument   
class ParametersCommon:
    """
    Class holding low level parameters of deap algorithm, parameters common to all environments
    
            selector : selector used in algorithm

            statistics : statistics deap will follow during the algorithm

            logbook_parameters : evolution records as chronological list of dictionnaries (optional)

            hall_of_fame_parameters : parameters to use for the hall-of-fame
    """

    selector={
        "function":tools.selBest,
        "parameters":{
        }
    }

    statistics={
        "parameters":{
            "key":lambda ind: ind.fitness.values
        },
        "to_register":{
            "avg": lambda x: np.mean(x, axis=0),
            "std": lambda x: np.std(x, axis=0),
            "min": lambda x: np.min(x, axis=0),
            "max": lambda x: np.max(x, axis=0),
            # "fit": lambda x: x,
        },
    }

    logbook_parameters={
    }

    hall_of_fame_parameters={
        "maxsize":1,
    }


class ParametersNN(ParametersCommon):
    """
    Class holding low level parameters of deap algorithm

            generator, breeder, mutator, selector : functions to use for corresponding deap 
                operators, define function then its parameters

            statistics : statistics deap will follow during the algorithm

            logbook_parameters : evolution records as chronological list of dictionnaries (optional)

            hall_of_fame_parameters : parameters to use for the hall-of-fame
    """

    generator={
        "function":lambda x, **kw: x(**kw),
        "parameters":{
            "input_dim":2, 
            "output_dim":2,
            "hidden_layers":3, 
            "hidden_dim":10,
            "use_batchnorm":False,
            "non_linearity":torch.tanh, 
            "output_normalizer":lambda x: torch.round((40 - 1) * abs(x)).int(),
        }
    }

    # breeder={
    #     "function":tools.cxSimulatedBinary,
    #     "parameters":{
    #         "eta":15,
    #     }
    # }
    breeder=None

    mutator={
        "function":tools.mutPolynomialBounded,
        "parameters":{
            "eta":15,
            "low":-1,
            "up":1,
            "indpb":.3,
        }
    }


class ParametersGuesser(ParametersCommon):
    """
    Class holding low level parameters of deap algorithm

            generator, breeder, mutator : functions to use for corresponding deap 
                operators, define function then its parameters
    """

    generator={
        "function":lambda x, **kw: x(**kw),
        "parameters":{
            "grid_size":20
        }
    }

    mutator={
        "function": lambda x, **kw: x.mutate(),
        "parameters":{
        }
    }

    breeder=None