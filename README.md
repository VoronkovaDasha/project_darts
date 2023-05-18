# Project: Differentiable Architecture Search

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
