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
from fed_core import get_dataloaders

class FedFisherClient:
    """
    Federated Learning Client that computes Fisher Information Matrix during local updates.
    """
    def __init__(self, cid, train_loader, device, lr, init_params):
        """
        Initialize the FedFisherClient instance.
        Args:
            cid (int): Client ID.
            train_loader (DataLoader): DataLoader for the client's training data.
            device (torch.device): Device to run the model on.
            lr (float): Learning rate for local training.
            init_params (dict): Initial model parameters for delta calculations.
        """
        self.cid = cid
        self.train_loader = train_loader
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

    def local_update(self, num_updates):
        """
        Perform local updates and compute the Fisher Information Matrix.
        Args:
            num_updates (int): Number of local update steps to perform.
        Returns:
            tuple: (state_dict, fisher_info, avg_loss)
                - state_dict (dict): Updated model parameters on CPU.
                - fisher_info (dict): Diagonal Fisher Information Matrix.
                - avg_loss (float): Average training loss over local updates.
        """
        self.model.train()
        losses = []

        # Initialize Fisher Matrix accumulators (diagonal only)
        fisher_diag = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher_diag[n] = torch.zeros_like(p, device=self.device)

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

            # Capture gradients for Fisher Information
            # F_i = E[grad^2]. Sum grad^2 here and average later.
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        fisher_diag[n] += p.grad.pow(2)

            self.optimizer.step()
            losses.append(loss.item())

        # Normalize Fisher (divide by N steps) and move to CPU
        final_fisher = {}
        for n in fisher_diag:
            avg_fisher = (fisher_diag[n] / len(losses)) + 1e-8
            final_fisher[n] = avg_fisher.cpu()

        # Return weights AND Fisher Information
        cpu_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        return cpu_state_dict, final_fisher, np.mean(losses) if losses else 0.0

##### Fisher Aggregation Logic #####
def fisher_aggregate(global_model, client_weights, client_fishers):
    """
    Performs precision-weighted averaging.
    w_new = (Sum F_i * w_i) / (Sum F_i)
    """
    with torch.no_grad():
        avg_state = {}
        total_fisher = {}

        # Initialize accumulators with zeros
        first_client_keys = client_weights[0].keys()
        for key in first_client_keys:
            # Shape check
            sample_tensor = client_weights[0][key]
            avg_state[key] = torch.zeros_like(sample_tensor, dtype=torch.float)
            total_fisher[key] = torch.zeros_like(sample_tensor, dtype=torch.float)

        # Accumulate weighted sums
        for i in range(len(client_weights)):
            w_i = client_weights[i]
            f_i = client_fishers[i]

            for key in w_i.keys():
                if key in f_i:
                    # Use Fisher weight
                    fisher_weight = f_i[key]
                else:
                    # Fallback small weight if Fisher not available
                    fisher_weight = (
                        torch.ones_like(w_i[key]) * 1e-5
                    )
                # Weighted sum
                avg_state[key] += w_i[key].float() * fisher_weight
                total_fisher[key] += fisher_weight

        # Normalize
        for key in avg_state.keys():
            avg_state[key] = avg_state[key] / total_fisher[key]

        global_model.load_state_dict(avg_state)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--local_updates", type=int, default=1)
    parser.add_argument("--config", type=str, default="pure_ssm_1_layer")
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_freq", type=int, default=30)
    parser.add_argument("--data_dir", type=str, default="../data/results/")
    parser.add_argument("--cache_dir", type=str, default="../../cache/wikitext2")
    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = get_device()

    # Experiment directories
    exp_name = (
        f"fed_fisher_config_{args.config}_lu_{args.local_updates}_seed_{args.seed}"
    )
    exp_dir = os.path.join(args.data_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    os.makedirs(f"{exp_dir}/server/params", exist_ok=True)
    os.makedirs(f"{exp_dir}/server/deltas", exist_ok=True)

    logging.info(f"Experiment: {exp_name} (Method: Fisher Aggregation)")

    # Load data
    client_train_loaders, global_val_loader, tokenizer = get_dataloaders(
        args.num_clients, args.sequence_length, args.batch_size, args.cache_dir
    )
    
    # Initialize global model, save initial params
    config = ssm_config.configs[args.config]
    config.vocab_size = tokenizer.vocab_size
    global_model = MambaLMHeadModel(config).to(device)
    init_params = get_ssm_params(global_model)

    clients = [
        FedFisherClient(i, client_train_loaders[i], device, args.lr, init_params)
        for i in range(args.num_clients)
    ]

    history = []
    global_step = 0
    local_steps_per_round = args.local_updates

    # Run training
    for epoch in range(args.n_epochs):
        # Reset clients' models to global model at start of each epoch
        for client in clients:
            client.reset_model(global_model.state_dict(), args.config)

        round_pbar = tqdm(
            range(
                len(client_train_loaders[0]) // args.batch_size // local_steps_per_round
            ),
            desc=f"Epoch {epoch+1}/{args.n_epochs}",
        )

        # Rounds of federated training
        for round_step in round_pbar:
            global_step += 1

            client_weights = []
            client_fishers = []
            client_losses = []

            # Local updates on each client, Fisher computation
            for client in clients:
                local_state, local_fisher, avg_loss = client.local_update(
                    local_steps_per_round
                )
                client_weights.append(local_state)
                client_fishers.append(local_fisher)
                client_losses.append(avg_loss)

            # Fisher Aggregation
            fisher_aggregate(global_model, client_weights, client_fishers)

            train_ppl = np.exp(np.mean(client_losses))
            round_pbar.set_postfix({"train_ppl": f"{train_ppl:.2f}"})

            # Evaluate and save at specified frequency
            if global_step % args.val_freq == 0 or global_step == 1:
                val_loss = evaluate(global_model, global_val_loader, device)
                val_ppl = np.exp(val_loss)

                # Save server params
                curr_params = get_ssm_params(global_model)
                save_npz(
                    f"{exp_dir}/server/params/step_{global_step:06d}.npz", curr_params
                )
                
                gen_text = generate_text(global_model, tokenizer, device=device)

                record = {
                    "global_step": global_step,
                    "train_ppl": float(train_ppl),
                    "server_val_ppl": float(val_ppl),
                    "generated": gen_text,
                }
                history.append(record)

                logging.info(
                    f"Step {global_step:4d} | Train PPL: {train_ppl:.2f} | Server Val PPL: {val_ppl:.2f}"
                )

    save_json(history, f"{exp_dir}/training_history.json")
    logging.info(f"Fisher Federated training complete. Results saved to {exp_dir}")
