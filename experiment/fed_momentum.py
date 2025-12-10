# fed_momentum.py
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

class FedMomentumClient:
    """
    Federated Learning Client with Momentum Aggregation    
    """
    def __init__(self, cid, train_loader, device, lr, init_params):
        """
        Initialize the FedMomentumClient instance.
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
        Perform local training updates.
        Args:
            num_updates (int): Number of local updates to perform.
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

        return self.model.state_dict(), np.mean(losses) if losses else 0.0

##### Momentum server aggregation logic #####
class MomentumServer:
    def __init__(self, model, momentum=0.9, learning_rate=1.0):
        """
        Initializes the Momentum Server Optimizer. Velocity is initialized to zero.
        Args:
            model (torch.nn.Module): The global model to be updated.
            momentum (float): Momentum factor (beta).
            learning_rate (float): Step size for the server update.
        """
        self.model = model
        self.momentum = momentum
        self.server_lr = learning_rate
        self.velocity = {k: torch.zeros_like(p) for k, p in model.state_dict().items()}

    def step(self, client_weights):
        """
        Applies Server-Side Momentum
        1. Compute pseudo-gradient: G = (w_old - w_avg_clients)
        2. Update velocity: v_new = beta * v_old + G
        3. Update global: w_new = w_old - server_lr * v_new
        Args:
            client_weights (list): List of state_dicts from clients after local updates.
        """
        # Compute simple average of clients (Measurement)
        avg_weights = {}
        with torch.no_grad():
            for key in client_weights[0].keys():
                avg_weights[key] = torch.stack(
                    [w[key].float() for w in client_weights]
                ).mean(0)

        # Update Global Model
        current_weights = self.model.state_dict()
        new_state_dict = {}
        
        # For each parameter, compute direction, update velocity, and update weights
        for key in current_weights.keys():
            direction = avg_weights[key] - current_weights[key].float()

            # Update velocity 
            self.velocity[key] = (
                self.momentum * self.velocity[key] + (1.0 - self.momentum) * direction
            )

            # Update weights
            new_state_dict[key] = (
                current_weights[key].float() + self.server_lr * self.velocity[key]
            )

        self.model.load_state_dict(new_state_dict)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Parse arguments
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
    # New Momentum Args
    parser.add_argument(
        "--server_momentum", type=float, default=0.9, help="Beta for Momentum velocity"
    )
    parser.add_argument(
        "--server_lr", type=float, default=1.0, help="Step size for server update"
    )
    parser.add_argument("--data_dir", type=str, default="../data/results/")
    parser.add_argument("--cache_dir", type=str, default="../../cache/wikitext2")
    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = get_device()

    # Experiment directories
    exp_name = (
        f"fed_momentum_config_{args.config}_lu_{args.local_updates}_seed_{args.seed}"
    )
    exp_dir = os.path.join(args.data_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/server/params", exist_ok=True)

    logging.info(f"Experiment: {exp_name} (Method: Momentum)")

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
        FedMomentumClient(i, client_train_loaders[i], device, args.lr, init_params)
        for i in range(args.num_clients)
    ]

    # Initialize Momentum Server
    server_optimizer = MomentumServer(
        global_model, momentum=args.server_momentum, learning_rate=args.server_lr
    )

    history = []
    global_step = 0
    local_steps_per_round = args.local_updates

    # Run training
    for epoch in range(args.n_epochs):
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
            client_losses = []

            for client in clients:
                local_state, avg_loss = client.local_update(local_steps_per_round)
                client_weights.append(local_state)
                client_losses.append(avg_loss)

            # Momentum Server Update
            server_optimizer.step(client_weights)

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
                    "generated": gen_text,
                }
                history.append(record)

                logging.info(f"Step {global_step:4d} | PPL: {val_ppl:.2f}")

    save_json(history, f"{exp_dir}/training_history.json")
    logging.info(f"Momentum Federated training complete. Results saved to {exp_dir}")
