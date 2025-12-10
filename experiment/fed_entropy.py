# fed_entropy.py
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import logging
import argparse
import os
import numpy as np
from tqdm import tqdm
import ssm_config
from utils import (
    set_seed,
    get_device,
    save_npz,
    save_json,
    get_ssm_params,
    evaluate,
    generate_text,
    init_mamba,
)
from fed_core import get_entropy_dataloaders

class FedEntropyClient:
    """ 
    Federated Learning Client with Entropy-based Aggregation
    """
    def __init__(self, cid, train_loader, val_loader, device, lr, init_params):
        """
        Initialize the FedEntropyClient instance.
        Args:
            cid (int): Client ID.
            train_loader (DataLoader): DataLoader for the client's training data.
            val_loader (DataLoader): DataLoader for the client's validation data.
            device (torch.device): Device to run the model on.
            lr (float): Learning rate for local training.
            init_params (dict): Initial model parameters for delta calculations.
        """
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = None
        self.optimizer = None
        self.init_params = init_params
        self.dataloader_iter = None
        self.lr = lr

    def reset_model(self, global_state_dict, config):
        """
        Reset the local model to the global model state.
        Args:
            global_state_dict (dict): State dictionary of the global model.
            config (str): Configuration name for initializing the model.
        """
        if self.model is None:
            self.model = init_mamba(config_name=config, vocab_size=50257).to(self.device)
        self.model.load_state_dict(global_state_dict)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )
        self.dataloader_iter = iter(self.train_loader)

    def calculate_entropy(self, max_batches=5):
        """Calculate the average entropy of the 
        predictive distribution on validation data.
         Args:
             max_batches (int): Maximum number of batches to evaluate for entropy.
         Returns:
             float: Average entropy over evaluated batches.
         """
        self.model.eval()
        entropies = []
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= max_batches:
                    break
                inputs = batch["input_ids"].to(self.device)
                logits = self.model(inputs).logits

                # Softmax to get probabilities: p(x)
                probs = torch.softmax(logits, dim=-1)
                # Entropy: - sum(p(x) * log(p(x)))
                batch_entropy = -torch.sum(
                    probs * torch.log(probs + 1e-8), dim=-1
                ).mean()
                entropies.append(batch_entropy.item())

        self.model.train()
        return (
            np.mean(entropies) if entropies else 10.0
        )  # Default high entropy if none computed

    def local_update(self, num_updates):
        """
        Run local training for a specified number of updates.
        Args:
            num_updates (int): Number of local updates to perform.
        Returns:
            tuple: Updated model state dictionary, entropy score, and average training loss.
        """
        self.model.train()
        losses = []
        for _ in range(num_updates):
            try:
                batch = next(self.dataloader_iter)
            except StopIteration:
                self.dataloader_iter = iter(self.train_loader)
                batch = next(self.dataloader_iter)

            inputs = batch["input_ids"].to(self.device)
            targets = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs).logits[:, :-1, :].contiguous()
            loss = nn.CrossEntropyLoss()(
                logits.contiguous().view(-1, logits.size(-1)),
                targets[:, 1:].contiguous().view(-1),
            )
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        # Calculate entropy score after training
        entropy_score = self.calculate_entropy()

        return (
            self.model.state_dict(),
            entropy_score,
            np.mean(losses) if losses else 0.0,
        )


##### Entropy-based Aggregation Function #####
def entropy_aggregate(global_model, client_weights, client_entropies, temperature=1.0):
    """
    Aggregates weights based on Softmin of entropy.
    alpha_k = exp(-H_k / T) / sum(exp(-H_j / T))
    Args:
        global_model (torch.nn.Module): The global model to update.
        client_weights (list): List of state_dicts from clients.
        client_entropies (list): List of entropy scores from clients.
        temperature (float): Temperature parameter for softmax.
    """
    with torch.no_grad():
        # Calculate weights from entropies
        entropies_tensor = torch.tensor(client_entropies)
        # Softmax(-H) = Softmin(H)
        weights = torch.softmax(-entropies_tensor / temperature, dim=0)

        logging.info(f"Aggregation Weights: {weights.tolist()}")  # Log who got trusted

        avg_state = {}
        first_client_keys = client_weights[0].keys()

        # Weighted Sum
        for key in first_client_keys:
            # Initialize with first client * their weight
            avg_state[key] = client_weights[0][key].float() * weights[0]

            # Add remaining
            for i in range(1, len(client_weights)):
                avg_state[key] += client_weights[i][key].float() * weights[i]

        global_model.load_state_dict(avg_state)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--local_updates", type=int, default=1)
    parser.add_argument("--config", type=str, default="pure_ssm_1_layer")
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_freq", type=int, default=30)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax weighting",
    )
    parser.add_argument("--data_dir", type=str, default="../data/results/")
    parser.add_argument("--cache_dir", type=str, default="../../cache/wikitext2")
    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = get_device()

    # Experiment directories
    exp_name = (
        f"fed_entropy_config_{args.config}_lu_{args.local_updates}_seed_{args.seed}"
    )
    exp_dir = os.path.join(args.data_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/server/params", exist_ok=True)

    logging.info(f"Experiment: {exp_name} (Method: Entropy Aggregation)")
    
    # Load data
    client_train_loaders, client_val_loaders, global_val_loader, tokenizer = (
        get_entropy_dataloaders(
            args.num_clients, args.sequence_length, args.batch_size, args.cache_dir
        )
    )

    # Initialize global model, save initial params
    config = ssm_config.configs[args.config]
    config.vocab_size = tokenizer.vocab_size
    global_model = MambaLMHeadModel(config).to(device)
    init_params = get_ssm_params(global_model)

    clients = [
        FedEntropyClient(
            i,
            client_train_loaders[i],
            client_val_loaders[i],
            device,
            args.lr,
            init_params,
        )
        for i in range(args.num_clients)
    ]

    history = []
    global_step = 0
    local_steps_per_round = args.local_updates

    # Run training
    for epoch in range(args.n_epochs):
        # At the beginning of each epoch, reset each client's model to the global model
        for client in clients:
            client.reset_model(global_model.state_dict(), args.config)

        round_pbar = tqdm(
            range(
                len(client_train_loaders[0]) // args.batch_size // local_steps_per_round
            ),
            desc=f"Epoch {epoch+1}/{args.n_epochs}",
        )

        # Federated rounds
        for round_step in round_pbar:
            global_step += 1
            client_weights = []
            client_entropies = []
            client_losses = []

            # Run local updates on each client, calculate weights, entropies, and losses
            for client in clients:
                local_state, entropy, avg_loss = client.local_update(
                    local_steps_per_round
                )
                client_weights.append(local_state)
                client_entropies.append(entropy)
                client_losses.append(avg_loss)

            # Aggregate using Entropy-based weights
            entropy_aggregate(
                global_model, client_weights, client_entropies, args.temperature
            )

            train_ppl = np.exp(np.mean(client_losses))
            round_pbar.set_postfix({"train_ppl": f"{train_ppl:.2f}"})

            # Evaluate and save params/deltas at specified frequency
            if global_step % args.val_freq == 0 or global_step == 1:
                val_loss = evaluate(global_model, global_val_loader, device)
                val_ppl = np.exp(val_loss)

                curr_params = get_ssm_params(global_model)
                save_npz(
                    f"{exp_dir}/server/params/step_{global_step:06d}.npz", curr_params
                )

                gen_text = generate_text(global_model, tokenizer, device=device)

                record = {
                    "global_step": global_step,
                    "train_ppl": float(train_ppl),
                    "server_val_ppl": float(val_ppl),
                    "avg_client_entropy": float(np.mean(client_entropies)),
                    "generated": gen_text,
                }
                history.append(record)

                logging.info(
                    f"Step {global_step:4d} | PPL: {val_ppl:.2f} | Avg Entropy: {np.mean(client_entropies):.4f}"
                )

    save_json(history, f"{exp_dir}/training_history.json")
    logging.info(f"Entropy Federated training complete. Results saved to {exp_dir}")
