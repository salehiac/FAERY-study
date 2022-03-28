

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

import sys
X=np.load(sys.argv[1])["arr_0"]
num_appended_population_priors=int(sys.argv[2]) #get this from embedding_info.txt, if this file isn't present then it is 0 

if num_appended_population_priors>0:
    appended_pop=X[X.shape[0]-num_appended_population_priors:,:]
    X=X[:X.shape[0]-num_appended_population_priors]
else:
    appended_pop=None


bandwidth = estimate_bandwidth(X, quantile=0.05, n_samples=3000)
#bandwidth=15
print("estimated bandwidth==",bandwidth)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#ms = MeanShift(bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

single_color=True
if not single_color:
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    colors = cycle('b')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.', label="solvers")
        #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
        #         markeredgecolor='k', markersize=14)
else:
    plt.plot(X[:, 0], X[:, 1], "bo", label="solvers")
    if appended_pop is not None:
        plt.plot(appended_pop[:,0],appended_pop[:,1],"ro",label="learned population priors")

plt.gca().set_aspect('equal', adjustable='box');
#plt.title("h",fontsize=28)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.xlabel("$e_1$",fontsize=28)
plt.ylabel("$e_2$",fontsize=28)
#plt.legend(fontsize=21)
plt.show()
