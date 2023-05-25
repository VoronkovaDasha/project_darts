# DA060492 Course Project: Differentiable Architecture Search

This repsitory contains the source code to reproduce the experiments from the final report on DA060492 Course Project: Differentiable Architecture Search.

## Project Description

Neural Architecture Search is a part of the AutoML field where a large number of various algorithms have emerged in recent years. One of the remarkable approaches is DARTS [Liu, H., Simonyan, K., and Yang., Y. Darts: Differentiable
architecture search. ICLR, 2019.], differentiable architecture search. In this work, the authors managed to relax the discrete search space to the continuous one and applied gradient-based optimization to architecture cell search. The models learned with DARTS achieve competitive performance on several tasks while requiring sufficiently reduced computational cost.

The goal of this project is to implement the DARTS method and evaluate the learned model. More precisely, the project objectives include the choice of an appropriate convolutional architecture and a reasonable search space, implementation bilevel optimization and evaluation of the learned architecture on a dataset of small size (e.g. CIFAR10).

## Overview of Results

First, we learn the architectures of normal and reduction cells using our implementation of DARTS approach on CIFAR10 dataset with both first- and second-order approximation. 
Second, we compare the performance and computational costs of the learned cells for first- and second-order approximation. Cells learned with first-order approximation achieve 71.42% validation accuraccy vs 74.64% validation accuracy for cells learned with second-order approximation.
Third, we compare performance of the final network architecures trained on CIFAR10 for both types of learned cells. Cells learned with first-order approximation achieve 78.28% test accuraccy vs 79.67% for cells learned with second-order approximation.
Finally, we compare results for the tranfer of the learned cells to CIFAR100 dataset. We obtain 38.45% test accuracy for first-order approximation and 46.41% for second-order approximation.

## Content

- DARTS_project.pdf - report for assignment "First project status report";
- DARTS_project_second_report.pdf - report for assignment "Second project status report";
- DARTS_project_final_project_submission.pdf - report for assignment "Final projectst submission";
- first_project_peer_review.pdf - review for the assignemnt "First project peer review";
- bilevel_optimization_example.ipynb - example of bilevel optimization;
- operations_cell_network.py - script with set of operations, cell architecture and network architecture;
- train_arch.py - script to perform architecture search;
- train_network.py - script to train final network architecture.

## How to run

1. To train cell architecture, run: python3 train_arch.py --n_epochs 40 --use_xi
- --n_epochs - number of epochs to train;
- --use_xi - perform second-order approximation (omit for first-order approximation).
During training, training and validation losses and accuracies are saved to train_stats.csv.
Also, during training, the values of $\alpha_{\text{normal}}$, $\alpha_{\text{reduce}}$ are stored in the .npy files (alpha_normal_history.npy, alpha_reduce_history.npy).
All files are saved to the current working directory.

2. To train the final network architecture, run: python3 train_network.py --dataset CIFAR10 --n_epochs 600 --alpha_normal_path 'alpha_normal_history.npy' --alpha_reduce_path 'alpha_reduce_history.npy' --alpha_epoch 40 --batch_size 64
- --dataset - dataset to use, CIFAR10 or CIFAR100;
- --n_epochs - number of epochs to train;
- --alpha_normal_path - path to file "alpha_normal_history.npy" from step 1;
- --alpha_reduce_path - path to file "alpha_reduce_history.npy" from step 1;
- --alpha_epoch - number of chekpoint for alpha to use;
- --batch_size - batch_size to use.
