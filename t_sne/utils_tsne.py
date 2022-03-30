import os
import pickle


# NEED CHANGE PATH


def get_files(path, basename):
    """
    Returns all path's basename content as list
    """

    base_list = []
    for root, dirs, files in os.walk("."):
        
        for name in files:
            
            if basename in name:
                with open(os.path.join(root, name), 'rb') as f:
                    base_list += pickle.load(f)
    
    return base_list


def get_solvers(path):
    """
    Returns all paths' solvers' behavior descriptors as list
    """

    return get_files(path, "solvers_")


def get_parameters(path):
    """
    Returns all paths' solvers' behavior descriptors as list
    """

    return get_files(path, "population_gen_")