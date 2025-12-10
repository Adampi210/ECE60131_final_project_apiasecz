# ECE60131 Final Project: Federated Learning for Selective State-Space Models

## 1. Project Overview & Methodology
### Abstract
For this project, I explored the application of **Federated Learning (FL)** to **Selective State-Space Models (SSMs)**, specifically focusing on the **Mamba2** architecture. The goal was to investigate the performance and feasibility of training SSMs in a distributed manner across multiple clients, enabling parallel learning while preserving data privacy.

The objective is to analyze the convergence behavior of SSMs when data is decentralized and to evaluate different aggregation strategies to see if they can improve the learning process.

### Key Contributions
1.  **Full-Parameter Federated Mamba2 Training:** I implemented a custom FL framework to train Mamba2-based custom language models (1-layer, 2-layer, and 4-layer configurations) on the WikiText-2 dataset without using adapters or LoRA.
2.  **Advanced Aggregation Algorithms:** Beyond standard FedAvg, I implemented and evaluated three distinct aggregation strategies:
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
