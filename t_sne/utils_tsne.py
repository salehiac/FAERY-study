import os
import pickle


def get_solvers(path):
    """
    Returns all paths' solvers' behavior descriptors as list
    """

    solvers = []
    for root, dirs, files in os.walk(os.path.expanduser(path)):

        for name in files:
            if "solvers_" in name:
                with open(os.path.join(root, name), 'rb') as f:
                    solvers += pickle.load(f)
    
    return solvers