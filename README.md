# FAERY

## Introduction

An implementation of the *FAERY* [[1]](#1) meta-learning algorithm for divergent search to the *Metaworld* benchmark [[2]](#2). Consists in a multi-objective optimization of a meta-population evaluated in a given number of evoluationary processes ran on parallelized environments.

For a meta-individual $p_j$ evaluated on $M$ environments in which it produced the $\zheta_{ij}$ lineage, it is given two scores :

$$
f_0(p_j) = \sum_{i=0}^{M-1}\sum_{s\in\zeta_{ij}} \psi_i(s)
$$
$$
f_1(p_j) = \frac{-1}{f_0(p_j)}\sum_{i=0}^{M-1}\sum_{s\in\zeta_{ij}}\psi_i(s)d_m(s,p_j)
$$

with $\psi_i(s)$ a binary operator checking if $s$ solved the environment and $d_m(s, p_j)$ the expected number of mutations to reach $s$ from $p_j$ (approximated by its depth in the genealogy tree).

## File structure

We used a set of prefix to distinguish our many files :
    •*class_* for files implementing a set of classes,
    •*utils_* for sets of utilitary functions
    •*extract_* to extract results and reproduce our graphs

We roughly have the following structure :
    • Problem -> contains base classes for Metaworld
        ◦ 
        ◦ 
        ◦ 
    • At the bottom: 
        ◦ Alpha slider, changes the amount of entropy
        ◦ Transparency slider, changes the transparency of the surfaces for better readability

## How to run the implementation

Python 3.6+, [requirements](./requirements.txt)


## References
<a id="1">[1]</a> 
*Few-shot Quality-Diversity Optimization*, Salehi and Coninx and Doncieux, IEEE Robotics and Automation Letters [2022](https://ieeexplore.ieee.org/abstract/document/9705622)
<a id="1">[2]</a> 
*Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning*, Tianhe Yu and Deirdre Quillen and Zhanpeng He and Ryan Julian and Karol Hausman and Chelsea Finn and Sergey Levine, Conference on Robot Learning (CoRL) [2019](https://arxiv.org/abs/1910.10897)