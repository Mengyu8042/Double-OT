# OT for GRN Reconstruction with Unpaired Samples
This repository contains the implementation of our work **"Double Optimal Transport for Gene Regulatory Network Reconstruction with Unpaired Samples"**.

## Introduction
This section provides a brief overview of the folders and files included in this repository:
* `methods/`: Contains the implementation of existing GRN inference methods.
* `demo_w_competing.ipynb`: Demonstration notebook comparing our method with all competing methods discussed in the manuscript.
* `demo_wo_competing.ipynb`: Demonstration notebook comparing our method only with baseline methods, without including all competing methods.
* `double_ot.py`: Implementation of the proposed Double OT method.
* `plot_utils.py`: Contains useful functions for plotting.
* `utils.py`: Contains useful functions for data generation, processing, and result evaluation.

## Reproducibility
To reproduce the results presented in the manuscript (e.g., Figure 5), you can run `demo_w_competing.ipynb`. Please note that this script implements all the competing methods, so it will take several hours to complete. Additionally, it depends on the [MATLAB Engine](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html) and the [Python-R bridge (rpy2)](https://rpy2.github.io/).

If you prefer a quicker demonstration or if your platform does not support MATLAB or R, you can run `demo_wo_competing.ipynb` instead. This script only compares the Double OT method with baseline methods and will complete in a few minutes.
