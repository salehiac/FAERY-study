# FAERY

## Introduction

An implementation of the *FAERY* [[1]](#1) meta-learning algorithm for divergent search to the *Metaworld* benchmark [[2]](#2). Consists in a multi-objective optimization of a meta-population evaluated in a given number of evoluationary processes ran on parallelized environments.

For a meta-individual $p_j \text{ evaluated on } M \text{ environments in which it produced the }\zeta_{ij}$ lineage, it is given two scores :

![FAERY_scores](https://user-images.githubusercontent.com/49323355/176115634-593fc1aa-cc76-4ff2-8f28-49706c3e263b.png)

with $\psi_i(s)$ a binary operator checking if $s$ solved the environment and $d_m(s, p_j)$ the expected number of mutations to reach $s$ from $p_j$ (approximated by its depth in the genealogy tree).

## File structure

We used a set of prefix to distinguish our many files :
- *class_* for files implementing a set of classes
- *utils_* for sets of utilitary functions
- *extract_* to extract results and reproduce our graphs

We strive to offer a modular implementation of the algorithm, hence each folder implements a package of classes :
- [problem](./problem) implements Metaworld agents as well as problem samplers
- [novelty_search](./novelty_search) implements examples for the inner algorithm, here Novelty Search [[3]](#3) and Quality Diversity [[4]](#4).
- [meta_learning](./meta_learning) implements a base class for meta-learning algorithms, in particular and implementation of *FAERY*
- [gridworld](./gridworld) toy environment, standalone implementation of *FAERY*
- [home](./) contains files to extract results and plot graphs (not detailed yet..) and their subsequent .json parameters files

## How to run the implementation

Python 3.6+, [requirements](./requirements.txt)

We included an [example script](./launchers/FAERY.sh) able to launch *FAERY* with base parameters on different environments. It takes as argument the desired environment (assembly, basketball, buttonpress, hammer) and outputs the results in the [results folder](./results).

To run a quick test run the following command from the [FAERY](./) directory :
> ./launchers/FAERY.sh test

## References
<a id="1">[1]</a> 
A. Salehi, A. Coninx and S. Doncieux, "Few-Shot Quality-Diversity Optimization," in IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 4424-4431, [April 2022](https://ieeexplore.ieee.org/abstract/document/9705622), doi: 10.1109/LRA.2022.3148438.

<a id="2">[2]</a> 
*Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning*, Tianhe Yu and Deirdre Quillen and Zhanpeng He and Ryan Julian and Karol Hausman and Chelsea Finn and Sergey Levine, Conference on Robot Learning (CoRL) [2019](https://arxiv.org/abs/1910.10897)

<a id="3">[3]</a> 
Stephane Doncieux, Alban Laflaquière, Alexandre Coninx. Novelty search: a Theoretical Perspective.
GECCO ’19: Genetic and Evolutionary Computation Conference, [Jul 2019](https://hal.archives-ouvertes.fr/hal-02561846/document), Prague Czech Republic,
France. pp.99-106, 10.1145/3321707.3321752. hal-02561846

<a id="4">[4]</a> 
A. Cully and Y. Demiris, "Quality and Diversity Optimization: A Unifying Modular Framework," in IEEE Transactions on Evolutionary Computation, vol. 22, no. 2, pp. 245-259, [April 2018](https://ieeexplore.ieee.org/document/7959075), doi: 10.1109/TEVC.2017.2704781.
