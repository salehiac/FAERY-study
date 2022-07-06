# FAERY

## Introduction

An implementation of the *FAERY* [[1]](#1) meta-learning algorithm for divergent search to the *Metaworld* benchmark [[2]](#2). Consists in a multi-objective optimization of a meta-population evaluated in a given number of evoluationary processes ran on parallelized environments.

For a meta-individual $p_j \text{ evaluated on } M \text{ environments in which it produced the }\zeta_{ij}$ lineage, it is given two scores :

<p align="center">
  <img src="https://user-images.githubusercontent.com/49323355/176396428-6471f752-74da-4dfc-bc0c-52aacf7c84c6.PNG" width="500" />
</p>

<p align="center">
with $\psi_i(s)$ a binary operator checking if $s$ solved the environment and $d_m(s, p_j)$ the expected number of mutations to reach $s$ from $p_j$ (approximated by its depth in the genealogy tree).
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/49323355/176396447-7621cf09-b397-47ec-84bd-ecd839ba9fd1.PNG" width="500" /> 
</p>

## File structure

We strive to offer a modular implementation of the algorithm, hence each folder implements a package of classes :
- [problem](./problem) implements Metaworld agents as well as problem samplers
- [novelty_search](./novelty_search) implements examples for the inner algorithm, here Novelty Search [[3]](#3) and Quality Diversity [[4]](#4).
- [meta_learning](./meta_learning) implements a base class for meta-learning algorithms, in particular an implementation of *FAERY*
- [gridworld](./gridworld) toy environment, standalone implementation of *FAERY*, work in progress
- [results](./results) contains output data, as well as python scripts that extract results and plot graphs

## How to run the implementation

Python 3.6+, [requirements](./requirements.txt)

We included an [example script](./launchers/FAERY.sh) able to launch *FAERY* with base parameters on different environments. It takes as argument the desired environment (assembly, basketball, buttonpress, hammer, and others if you create the corresponding [base_{env}.params](./launchers/parameters) file) and outputs the results in the [results folder](./results).

To run a quick test and check that everything works, run the following command from the [FAERY](./) directory :
```bash
# To run a quick test:
./launchers/FAERY.sh test
# To run FAERY on Metaworld hammer:
./launchers/FAERY.sh hammer
```

During each meta-step, the solvers are saved as they are found in [results/data](./results/data). Then, the final evolution tables and meta-population's scores $f_0$ and $f_1$ are computed and saved for the given meta-step. Therefore, once the algorithm has ran, you should be able to extract and compile them as performance graphs or T-SNE plots by running the provided scripts in [extraction scripts](./results/) :

<p float="left">
  <img src="https://user-images.githubusercontent.com/49323355/176389285-8d4becf5-691e-4930-a28b-4bbbd1454c03.png" width="79%" />
  <img src="https://user-images.githubusercontent.com/49323355/176389404-8d320a08-4801-4562-ab92-a5ed415d7044.png" width="20%" /> 
</p>

## Set up the extraction

In order to plot the above graphs, one needs to set up the scripts with the provided [.json files](./results/), the parameters of which are self explanatory. Please note that by setting some booleans, it is possible to enable the plotting of supplementary graphs, such as :

- a comparison graph for data obtained with different parameters
<p align="center">
  <img src="https://user-images.githubusercontent.com/49323355/176408431-66fe85a5-924d-4772-9656-32adfc74e73e.png" width="500" /> 
</p>

- an animation of the pareto front
<p align="center">
  <img src="https://user-images.githubusercontent.com/49323355/176412005-c093e954-0224-4c09-bc84-e154a9a8ff05.gif" width="500" />
</p>

- a TSNE plot for different perplexity parameters
<p align="center">
  <img src="https://user-images.githubusercontent.com/49323355/176412242-f92ab1d3-9f58-46e7-826a-1e7ddb488478.png" width="100%" />
</p>

- an animation of the solutions within the TSNE plot as they are found
<p align="center">
  <img src="https://user-images.githubusercontent.com/49323355/176413285-5fb03a5c-d1f9-452a-98a8-357622debd05.gif" width="500" />
</p>

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
