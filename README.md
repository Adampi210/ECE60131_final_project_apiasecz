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

## 4. Execution Instructions
All training scripts are located in the experiment/ directory. Make sure to cd into this directory before running any scripts.

```bash
cd experiment
```

A. Centralized Baseline
Run the centralized training to establish a baseline for training performance.

```bash
python central.py
```
the following flags can be adjusted when running the centralized training script:
```--config```: Choose the Mamba configuration (e.g., pure_ssm_1_layer, pure_ssm_2_layer, pure_ssm_4_layer).

```--n_epochs```: Set the number of training epochs (default: 1).

```--batch_size```: Define the batch size for training (default: 16).

```--lr```: Specify the learning rate for the optimizer (default: 5e-4).

```--seed```: Set the random seed for reproducibility (default: 0).
```--sequence_length```: The length of input sequences used for tokenization and training (default: 256).

```--val_freq```: Frequency (in global steps) to evaluate validation loss and save parameter deltas (default: 10).

```--data_dir```: Directory where experiment results and logs will be saved (default: ../data/results/).
```--cache_dir```: Directory to cache the downloaded WikiText-2 dataset and tokenizer files (default: ../../cache/wikitext-2).