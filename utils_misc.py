# Novelty Search Library.
# Copyright (C) 2020 Sorbonne University
# Maintainer: Achkan Salehi (salehi@isir.upmc.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import subprocess
import os

import pickle
import random
import copy
import torch
import string

import deap.creator
import deap.base
import deap.tools

import numpy as np

from datetime import datetime
from functools import reduce


def get_current_time_date():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def rand_string(alpha=True, numerical=True):
    l2 = "0123456789" if numerical else ""
    return reduce(lambda x, y: x + y,
                  random.choices(string.ascii_letters + l2, k=10), "")


def bash_command(cmd: list):
    """
    cmd  list [command, arg1, arg2, ...]
    """
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    out, err = proc.communicate()
    ret_code = proc.returncode

    return out, err, ret_code


def create_directory(dir_basename,
                     remove_if_exists=True,
                     pid=True):

    while dir_basename[-1] == "/":
        dir_basename = dir_basename[:-1]
    while dir_basename[0] == "/":
        dir_basename = dir_basename[1:]   

    dir_path = dir_basename + "_" + str(os.getpid()) if pid is True else dir_basename

    # if os.path.exists(dir_path):
    #     if remove_if_exists:
    #         shutil.rmtree(dir_path)
    #     else:
    #         raise Exception("directory exists but remove_if_exists is False")

    os.makedirs(dir_path, exist_ok=True)
    # MULTITHREAD???
    with open(dir_path + "/creation_notification.txt", "w") as fl:
        fl.write("created on " + get_current_time_date() + "\n")

    return dir_path


def dump_pickle(fn, obj):
    if fn[-4:] != ".pkl":
        fn += ".pkl"
    with open(fn, "wb") as fl:
        pickle.dump(obj, fl)


class colors:
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    yellow = (255, 255, 51)


_non_lin_dict = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "selu": torch.selu,
    "leaky_relu": torch.nn.functional.leaky_relu
}


def identity(x):
    """
    because pickle and thus scoop don't like lambdas...
    """
    return x


class SmallEncoder1d(torch.nn.Module):

    def __init__(self,
                 in_d,
                 out_d,
                 num_hidden=3,
                 non_lin="relu",
                 use_bn=False):
        torch.nn.Module.__init__(self)

        self.in_d = in_d
        self.out_d = out_d

        hidden_dim = 3 * in_d
        self.mds = torch.nn.ModuleList([torch.nn.Linear(in_d, hidden_dim)])

        for i in range(num_hidden - 1):
            self.mds.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.mds.append(torch.nn.Linear(hidden_dim, out_d))

        self.non_lin = _non_lin_dict[non_lin]

    def forward(self, x):
        """
        x list
        """
        out = torch.Tensor(x)
        for md in self.mds[:-1]:
            out = self.non_lin(md(out))

        return self.mds[-1](out)

    def weights_to_constant(self, cst):
        with torch.no_grad():
            for m in self.mds:
                m.weight.fill_(cst)
                m.bias.fill_(cst)

    def weights_to_rand(self, d=5):
        with torch.no_grad():
            for m in self.mds:
                # m.weight.fill_(np.random.randn(m.weight.shape)*d)
                # m.bias.fill_(np.random.randn(m.weight.shape)*d)
                # pdb.set_trace()
                m.weight.data = torch.randn_like(m.weight.data) * d
                m.bias.data = torch.randn_like(m.bias.data) * d


def selRoulette(individuals, k, fit_attr=None, automatic_threshold=True):
    """
    Based on the deap function of the same name, but adapted to novelty with more complex behavior. The fit_attr argument is never used, but is here
    for retro-compatibility issues
    """

    individual_novs = [x._nov for x in individuals]

    if automatic_threshold:
        md = np.median(individual_novs)
        individual_novs = list(
            map(lambda x: x if x > md else 0, individual_novs))

    s_indx = np.argsort(individual_novs).tolist()[::-1]  # decreasing order
    sum_n = sum(individual_novs)
    chosen = []
    for i in range(k):
        u = random.random() * sum_n
        sum_ = 0
        for idx in s_indx:
            sum_ += individual_novs[idx]
            if sum_ > u:
                chosen.append(copy.deepcopy(individuals[idx]))
                break

    return chosen


def selBest(individuals, k, automatic_threshold=True):

    individual_novs = [x._nov for x in individuals]

    if automatic_threshold:
        md = np.median(individual_novs)
        individual_novs = list(
            map(lambda x: x if x > md else 0, individual_novs))

    s_indx = np.argsort(individual_novs).tolist()[::-1]  # decreasing order
    return [individuals[i] for i in s_indx[:k]]


# Deap sometimes creates problems with parallelism if those are not called in the global scope
deap.creator.create("Fitness2d", deap.base.Fitness,
                    weights=(1.0, 1.0,))
deap.creator.create("LightIndividuals", list,
                    fitness=deap.creator.Fitness2d, ind_i=-1)


class NSGA2:
    """
    wrapper around deap's selNSGA2
    """

    def __init__(self, k):
        # Deap sometimes creates problems with parallelism if those are not called in the __main__ script
        # deap.creator.create("Fitness2d",deap.base.Fitness,weights=(1.0,1.0,))
        # deap.creator.create("LightIndividuals",list,fitness=deap.creator.Fitness2d, ind_i=-1)

        self.k = k

    def __call__(self, individuals):

        light_pop = []
        for i, individual in enumerate(individuals):
            light_pop.append(deap.creator.LightIndividuals())
            light_pop[-1].fitness.setValues([individual._fitness, individual._nov])
            light_pop[-1].ind_i = i

        return list(map(
            lambda x: individuals[x.ind_i],
            deap.tools.selNSGA2(light_pop, self.k, nd="standard")
        ))


def make_networks_divergent(frozen, trained, frozen_domain_limits, iters):
    """
    frozen                         frozen network
    trained                        network whose weights are learnt
    frozen_domain_limits           torch tensor of shape N*2. The baehavior space is for now assumed to be an N-d cube,
                                   and frozen_domain_limits[i,:] is the lower and higher bounds along that dimension
    iters                          int 
    """
    LR = 1e-3
    optimizer = torch.optim.Adam(trained.parameters(), lr=LR)

    assert frozen.in_d == trained.in_d and frozen.out_d == trained.out_d, "dims mismatch"

    batch_sz = 32
    for it_i in range(iters):

        trained.train()
        frozen.eval()

        batch = torch.zeros(batch_sz, frozen.in_d)
        for d_i in range(frozen.in_d):
            batch[:, d_i] = torch.rand(batch_sz) * (
                frozen_domain_limits[d_i, 1] -
                frozen_domain_limits[d_i, 0]) + frozen_domain_limits[d_i, 0]

        optimizer.zero_grad()
        target = frozen(batch)
        pred = trained(batch)
        loss = ((target - pred)**2).sum(1)
        # because we want networks to diverge
        loss = (loss.mean()).clone() * -1

        loss.backward()
        optimizer.step()


def get_sum_of_model_params(mdl):

    x = [x.sum().item() for x in mdl.parameters() if x.requires_grad]

    return sum(x)


def get_path(parser=None, to_parse=True, default="./results.json", verbose=True):
    """
    Returns the queried path
    """

    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_params",
        type=str,
        help="path to the json file",
        default=default
    )

    path = parser.parse_args().path_params
    if path[-5:] != ".json":    path += ".json"

    if verbose is True:
        print("Loaded parameters from {}".format(path))

    return path if to_parse is True else parser
