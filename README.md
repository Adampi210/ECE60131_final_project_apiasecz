# ECE60131 Final Project: Federated Learning for Selective State-Space Models

## 1. Project Overview & Methodology
### Abstract
For this project, I explored the application of **Federated Learning (FL)** to **Selective State-Space Models (SSMs)**, specifically focusing on the **Mamba2** architecture. The goal was to investigate the performance and feasibility of training SSMs in a distributed manner across multiple clients, enabling parallel learning while preserving data privacy.

**Main Objective:** analyze the convergence behavior of SSMs when data is decentralized and evaluate different aggregation strategies to see if they can improve the learning process.

### Key Contributions
1.  **Full-Parameter Federated Mamba2 Training:** I implemented a custom FL framework to train from scratch Mamba2-based custom language models (1-layer, 2-layer, and 4-layer configurations) on the WikiText-2 dataset (without using adapters or LoRA).
2.  **Advanced Aggregation Algorithms:** Beyond standard FedAvg, I proposed, implemented, and evaluated three distinct aggregation strategies:
    * **FedMomentum:** Applies server-side momentum to smooth updates and accelerate convergence.
    * **FedEntropy:** Uses an uncertainty-aware aggregation mechanism where client weights are scaled based on the entropy of their predictive distribution.
    * **FedFisher:** Implements a precision-weighted aggregation using the diagonal of the Fisher Information Matrix (FIM) computed locally by clients.
3.  **Performance Analysis:** I demonstrated that a gap persists between centralized and federated strategies, and that different aggregation methods result in varying performance. Notably, the **Fisher Information weighting** method outperformed standard FedAvg in terms of validation perplexity.

For details, see the included presentation slides. 

### Directory Structure
```text
.
├── README.md
├── ECE60131_final_project_presentation.pptx # Presentation slides
├── ai_default.yml          # Conda environment configuration
├── data
│   ├── plots               # Generated figures (heatmaps, perplexity curves)
│   └── results             # Raw experiment logs (.json) and model checkpoints (.npz)
├── experiment              # Source code for training
│   ├── central.py          # Centralized training baseline
│   ├── fed_avg.py          # Standard Federated Averaging
│   ├── fed_core.py         # Shared data loading logic for federated experiments
│   ├── fed_entropy.py      # Entropy-based aggregation
│   ├── fed_fisher.py       # Fisher Information aggregation
│   ├── fed_momentum.py     # Momentum-based aggregation
│   ├── ssm_config.py       # Mamba2 model configurations
│   └── utils.py            # Helper functions (seed, I/O, evaluation)
└── plot                    # Visualization scripts
    └── plot_perplexities.py
    └── visualize_federated_deltas.py
    └── visualize_deltas.py
```

## 2. Environment Setup (Conda)
To ensure reproducibility, follow these steps to set up the environment using Anaconda. Note that an internet connection is required for the initial setup to download the WikiText-2 dataset and install dependencies.

### Prerequisites
- Anaconda or Miniconda installed.

### Environment Installation Steps
1. **Create the Conda Environment:**
   Use the provided `ai_default.yml` file to create a new conda environment with all necessary dependencies:
```bash
conda env create -f ai_default.yml
```

2. **Activate the environment:**
```bash
conda activate ai_default
```

3. **Verify Mamba installation:** Ensure mamba_ssm and torch are correctly installed and accessible within the environment.

## 3. Dataset Preparation
This project uses the **WikiText-2 dataset** (raw version).

Automatic Download: There is no need to manually download the dataset. The code is designed to automatically download and cache the dataset via the Hugging Face datasets library upon the first execution.

Caching: Data is cached locally (default: ../../cache/wikitext2) to speed up subsequent runs.

## 4. Code Execution Instructions
All training scripts are located in the experiment/ directory. Make sure to cd into this directory before running any scripts.

```bash
cd experiment
```

### A. **Centralized Baseline**

Run the centralized training as a baseline for training performance.

```bash
python central.py
```
The following flags can be adjusted when running the centralized training script:

```--config```: Choose the Mamba configuration (e.g., pure_ssm_1_layer, pure_ssm_2_layer, pure_ssm_4_layer).

```--n_epochs```: Set the number of training epochs (default: 1).

```--batch_size```: Define the batch size for training (default: 16).

```--lr```: Specify the learning rate for the optimizer (default: 5e-4).

```--seed```: Set the random seed for reproducibility (default: 0).

```--sequence_length```: The length of input sequences used for tokenization and training (default: 256).

```--val_freq```: Frequency (in global steps) to evaluate validation loss and save parameter deltas (default: 10).

```--data_dir```: Directory where experiment results and logs will be saved (default: ../data/results/).
```--cache_dir```: Directory to cache the downloaded WikiText-2 dataset and tokenizer files (default: ../../cache/wikitext-2).

### B. **Standard Federated Averaging (FedAvg)**

Run the baseline federated learning algorithm. 
```bash
python fed_avg.py
```
The following flags can be adjusted when running the FedAvg training script:

```--num_clients```: Number of clients participating in federated learning (default: 4).

```--local_updates```: Number of local updates (steps) each client performs per communication round (default: 1).

```--config```: Choose the Mamba configuration (e.g., pure_ssm_1_layer, pure_ssm_2_layer, pure_ssm_4_layer).

```--n_epochs```: Number of global epochs (default: 1).

```--lr```: Learning rate for the optimizer (default: 5e-4).

```--batch_size```: Batch size for client training (default: 4).

```--sequence_length```: Length of input sequences for tokenization and training (default: 256).

```--seed```: Random seed for reproducibility (default: 0).

```--val_freq```: Frequency (in global steps) to evaluate validation loss and save parameter deltas (default: 30).

```--data_dir```: Directory where experiment results and logs will be saved (default: ../data/results/).

```--cache_dir```: Directory to cache the downloaded WikiText-2 dataset and tokenizer files (default: ../../cache/wikitext2).

### C. **Federated Learning with Momentum (FedMomentum)**

Uses server-side momentum to stabilize the global model updates:
```bash
python fed_momentum.py
```
The script has the same flags as fed_avg.py, with additional flags:

```--server_momentum```: Momentum coefficient for Momentum velocity (default: 0.9).

```--server_lr```: Step size for server update with Momentum (default: 1.0).

### D. **Federated Learning with Entropy Weighting (FedEntropy)**

Aggregates weights based on the confidence (entropy) of the client's model on validation data:
```bash
python fed_entropy.py
```
The script has the same flags as fed_avg.py, with an additional flag:

```--temperature```: Temperature parameter for entropy scaling (default: 1.0).

### E. **Federated Learning with Fisher Information Weighting (FedFisher)**

Computes the diagonal Fisher Information Matrix to weight parameters based on their importance during aggregation:
```bash
python fed_fisher.py
```
The script has the same flags as fed_avg.py.

## 5. Visualization & Analysis
After training, results (loss, perplexity, generated text) are saved in a corresponding directory in a designated results folder (default: ```../data/results/```). You can generate comparison plots using the scripts in the ```plot/``` directory.

Note: The results of the past experiments for perplexities are included in the ```data/results/``` directory. For delta visualization, the ```*.npz``` files are not uploaded (due to their size), and must be generated by running the training scripts first.

### A. Generating Perplexity Curves
This script generates comparisons between Central vs. Federated, the impact of local updates, and method comparisons (Fisher vs. Entropy vs. Momentum):
```bash
cd ../plot
python plot_perplexities.py --data_dir ../data/results/ --out_dir ../data/plots/perplexities --config pure_ssm_1_layer
```
The flags for this script are:

```--data_dir```: Directory containing experiment results and logs (default: ```../data/results/```).

```--out_dir```: Directory to save generated plots (default: ```../data/plots/perplexities```).

```--config```: Model configuration used for plotting results (except for last plot) (e.g., pure_ssm_1_layer, pure_ssm_2_layer, pure_ssm_4_layer).


**Output:** The script will generate the following plots in the out_dir:
- 1_central_vs_fed: Compares centralized training against FedAvg (LU=1) for the specified model configuration.
- 2_lu_comparison: Compares FedAvg with Local Updates = 1 vs. 8 for the specified model configuration.
- 3_method_comparison: Compares FedAvg, FedFisher, FedEntropy, and FedMomentum validation perplexity for a given model configuration.
- 4_multi_config_comparison: Compares different model depths (1, 2, 4 layers, all model configurations, FedAvg only).

### B. Visualizing Centralized Deltas (Heatmaps & SVD)
This script analyzes the parameter updates (deltas) for centralized experiments. It generates heatmaps of weight updates and plots the evolution of Singular Values (SVD) to understand the rank structure of learning.

```bash
python visualize_deltas.py --data_dir ../data/results/ --output_dir ../data/plots/deltas
```
The flags for this script are:

```--data_dir```: Root folder containing centralized experiment results (must match pattern central_*_seed_*).

```--output_dir```: Directory to save the heatmaps and SVD plots.

**Output:**
- ```heatmap_structure_*.png```: Visual grid of weight deltas across training steps for specific seeds.
- ```averaged_svd_evolution_*.png```: Evolution of the top singular values averaged across seeds.
- ```averaged_final_spectrum_*.png```: The final singular value spectrum showing the rank at 95% energy.

### C. Visualizing Federated Deltas (Heatmaps & SVD)
This script performs similar analysis for federated experiments as 5.B, separating analysis by method (FedAvg, Momentum, etc.) and Local Updates (LU).

```bash
python visualize_federated_deltas.py --data_dir ../data/results/ --output_dir ../data/plots/deltas --entity server
```

The flags for this script are:

```--data_dir```: Root folder containing federated experiment results.

```--output_dir```: Directory to save the plots.

```--entity```: Specify whether to visualize server deltas or client deltas (default: server).

**Output:**
- ```heatmap_*.png```: Heatmaps of parameter updates for specific federated runs.
- ```svd_evolution_*.png```: Evolution of singular values for the aggregated global model.
- ```svd_spectrum_*.png```: Final singular value spectrum for federated methods.

## 6. References
* **Mamba Architecture:** Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv preprint arXiv:2312.00752.
* **Federated Averaging (FedAvg):** McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.
* **Dataset:** Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). *Pointer Sentinel Mixture Models* (WikiText-2). ICLR.

## 7. Author
**Project Author:** Adam Piaseczny
