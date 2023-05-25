# DA060492 Course Project: Differentiable Architecture Search

This repsitory contains the source code to reeproduce the experiments from the final report on DA060492 Course Project: Differentiable Architecture Search.

## Project Description

Neural Architecture Search is a part of the AutoML field where a large number of various algorithms have emerged in recent years. One of the remarkable approaches is DARTS [Liu, H., Simonyan, K., and Yang., Y. Darts: Differentiable
architecture search. ICLR, 2019b.], differentiable architecture search. In this work, the authors managed to relax the discrete search space to the continuous one and applied gradient-based optimization to architecture cell search. The models learned with DARTS achieve competitive performance on several tasks while requiring sufficiently reduced computational cost.

The goal of this project is to implement the DARTS method and evaluate the learned model. More precisely, the project objectives include the choice of an appropriate convolutional architecture and a reasonable search space, implementation bilevel optimization and evaluation of the learned architecture on a dataset of small size (e.g. CIFAR10).


- DARTS_project.pdf - report for assignment "First project status report";
- DARTS_project_second_report.pdf - report for assignment "Second project status report";
- bilevel_optimization_example.ipynb - example of bilevel optimization;
- operations_cell_network.py - script with set of operations, cell architecture and network architecture;
- architecture_search.py - script to perform architecture search.

## How to run
python3 architecture_search.py --n_channels 16 --n_epochs 40 --use_xi
- --n_channels - number of channels for the first cell in the whole architecture;
- --n_epochs - number of epochs to train;
- --use_xi - whether to perform first-order or second-order optimization.

During training, training and validation losses and accuracies are saved to train_stats.csv.
Also, during training, the values of $\alpha_{\text{normal}}$, $\alpha_{\text{reduce}}$ are stored in the .npy files (alpha_normal_history.npy, alpha_reduce_history.npy).
All files are saved to the current working directory.
