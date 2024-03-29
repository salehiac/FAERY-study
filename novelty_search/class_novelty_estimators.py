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

import torch
import numpy as np

from abc import ABC, abstractmethod
from sklearn.neighbors import KDTree

import utils_misc


class NoveltyEstimator(ABC):
    """
    Interface for estimating Novelty
    """

    @abstractmethod
    def __call__(self):
        """
        estimate novelty of entire current population w.r.t istelf+archive
        """
        pass

        # getattr(_pop_bds[idx].
        # setattr(_pop_bds[idx], novelty_attr)   etc etc
    @abstractmethod
    def update(self, pop, archive=None):
        pass


class ArchiveBasedNoveltyEstimator(NoveltyEstimator):
    """
    """

    def __init__(self, k):
        self.k = k
        self.archive = None
        self.pop = None
        self.log_dir = "/tmp/"

    def update(self, pop, archive):
        self.archive = archive
        self.pop = pop

        self.pop_bds = np.concatenate([x._behavior_descr for x in self.pop], 0)
        self.archive_bds = [x._behavior_descr for x in self.archive]

        if len(self.archive_bds):
            self.archive_bds = np.concatenate(self.archive_bds, 0)
            self.kdt_bds = np.concatenate([self.archive_bds, self.pop_bds], 0)
        else:
            self.kdt_bds = self.pop_bds

        self.kdt = KDTree(self.kdt_bds, leaf_size=20, metric='euclidean')

    def __call__(self):
        """
        estimate novelty of entire current population w.r.t istelf+archive

        returns novelties as unsorted list
        """
        dists, ids = self.kdt.query(self.pop_bds, self.k, return_distance=True)
        # the first column is the point itself because the population itself is included in the kdtree
        dists = dists[:, 1:]
        ids = ids[:, 1:]

        novs = dists.mean(1)
        return novs.tolist()


class LearnedNovelty1d(NoveltyEstimator):

    def __init__(self,
                 in_dim,
                 emb_dim,
                 pb_limits=None,
                 batch_sz=128,
                 log_dir="/tmp/"):

        self.frozen = utils_misc.SmallEncoder1d(
            in_dim, emb_dim, num_hidden=3, non_lin="leaky_relu", use_bn=False
        )  # note that using batchnorm wouldn't make any sense here as the results of the frozen network shouldn't change depending on batch
        # self.frozen.weights_to_constant(1.0)
        # self.frozen.weights_to_rand(d=0.2)
        self.frozen.eval()

        self.learnt = utils_misc.SmallEncoder1d(in_dim,
                                               emb_dim,
                                               num_hidden=5,
                                               non_lin="leaky_relu",
                                               use_bn=False)

        #self.optimizer = torch.optim.SGD(self.learnt.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.learnt.parameters(), lr=1e-2)
        self.archive = None
        self.pop = None
        self.batch_sz = batch_sz

        self.epoch = 0

        self.log_dir = log_dir

        if pb_limits is not None:
            utils_misc.make_networks_divergent(
                self.frozen,
                self.learnt,
                pb_limits,
                iters=50
            )

    def update(self, pop, archive=None):

        self.pop = pop
        self.pop_bds = np.concatenate([x._behavior_descr for x in self.pop], 0)

    def __call__(self):

        # self.pop_bds is of size NxD with D the dimensions of the behavior space
        pop_novs = []
        for i in range(0, self.pop_bds.shape[0], self.batch_sz):
            batch = torch.Tensor(self.pop_bds[i:i + self.batch_sz])
            # batch=batch/600
            # batch=batch-0.5
            with torch.no_grad():
                # pdb.set_trace()
                e_frozen = self.frozen(batch)
                self.learnt.eval()
                e_pred = self.learnt(batch)
                diff = (e_pred - e_frozen)**2
                diff = diff.sum(1)
                pop_novs += diff.cpu().detach().tolist()

        assert len(pop_novs) == self.pop_bds.shape[0], "that shouldn't happen"

        return pop_novs

    def train(self, pop):

        if self.epoch == 0:
            torch.save(self.frozen.state_dict(),
                       self.log_dir + "/frozen_net.model")

        torch.save(self.learnt.state_dict(),
                   self.log_dir + f"/learnt_{self.epoch}.model")

        pop_bds = [x._behavior_descr for x in pop]
        pop_bds = np.concatenate(pop_bds, 0)
        for _ in range(3):
            for i in range(0, pop_bds.shape[0], self.batch_sz):
                batch = torch.Tensor(pop_bds[i:i + self.batch_sz])
                # batch=torch.Tensor(pop_bds[random.choices(range(len(self.pop)),k=(min(self.batch_sz,len(self.pop)))),:])
                # pdb.set_trace()
                # batch=torch.Tensor(pop_bds[])
                # batch=batch/600
                # batch=batch-0.5
                # pdb.set_trace()
                with torch.no_grad():
                    e_frozen = self.frozen(batch)

                self.learnt.train()
                self.optimizer.zero_grad()
                e_l = self.learnt(batch)
                ll = (e_l - e_frozen)**2
                ll = ll.sum(1)
                #weights=torch.Tensor([1/max(1,x._age+1) for x in pop])
                weights = torch.Tensor([1.0 for x in range(batch.shape[0])])
                loss = ll * weights
                loss = loss.mean().clone()
                # loss/=self.batch_sz
                if torch.isnan(loss).any():
                    raise Exception(
                        "loss is Nan. Maybe tray reducing the learning rate")
                loss.backward()
                self.optimizer.step()

        self.epoch += 1
