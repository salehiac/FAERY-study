# FAERY

## Introduction

An implementation of the *FAERY* meta-learning algorithm for divergent search (Few-shot Quality-Diversity Optimization, Salehi and Coninx and Doncieux, IEEE Robotics and Automation Letters [2022](https://ieeexplore.ieee.org/abstract/document/9705622))

Multi-objective optimization of a meta-population evaluated in a given number of evoluationary processes ran on parallelized environments.

For a meta-individual $p_j$ evaluated on $M$ environments in which it produced the $\zheta_{ij}$ lineage, it is given two scores :

$$
f_0(p_j) = \sum_{i=0}^{M-1}\sum_{s\in\zeta_{ij}} \psi_i(s) \\
f_1(p_j) = \frac{-1}{f_0(p_j)}\sum_{i=0}^{M-1}\sum_{s\in\zeta_{ij}}\psi_i(s)d_m(s,p_j)
$$

with $\psi_i(s)$ a binary operator checking if $s$ solved the environment; $d_m(s, p_j)$ the expected number of mutations to reach $s$ from $p_j$ $\approx$ depth in the genealogy tree.

## File structure



## How to run the implementation

Python 3.6+, [requirements](./requirements.txt)



