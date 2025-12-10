# fed_avg.py
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

class FedClient:
    """
    Base Federated Learning Client
    """
    def __init__(self, cid, train_loader, device, lr, init_params):
        """
        Initialize the FedClient instance.
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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.dataloader_iter = iter(self.train_loader)

    def local_update(self, num_updates):
        """
        Run local training for a specified number of updates.
        Args:
            num_updates (int): Number of local updates to perform.
        Returns:
            tuple: Updated model state dictionary and average training loss.
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


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument(
        "--local_updates", type=int, default=1, help="Local steps per round"
    )
    parser.add_argument("--config", type=str, default="pure_ssm_1_layer")
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--val_freq",
        type=int,
        default=30,
        help="Save params/deltas every N global steps",
    )
    parser.add_argument("--data_dir", type=str, default="../data/results/")
    parser.add_argument("--cache_dir", type=str, default="../../cache/wikitext2")
    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = get_device()

    # Experiment directories
    exp_name = f"federated_config_{args.config}_lu_{args.local_updates}_seed_{args.seed}"
    exp_dir = os.path.join(args.data_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/server/params", exist_ok=True)
    os.makedirs(f"{exp_dir}/server/deltas", exist_ok=True)
    for i in range(args.num_clients):
        os.makedirs(f"{exp_dir}/client_{i}/params", exist_ok=True)
        os.makedirs(f"{exp_dir}/client_{i}/deltas", exist_ok=True)

    logging.info(f"Experiment: {exp_name}")

    # Load data
    client_train_loaders, global_val_loader, tokenizer = get_dataloaders(
        args.num_clients, args.sequence_length, args.batch_size, args.cache_dir
    )

    # Initialize global model, save initial params
    global_model = init_mamba(args.config, tokenizer.vocab_size).to(device)
    init_params = get_ssm_params(
        global_model
    ) 
    save_npz(f"{exp_dir}/server/params/step_000000.npz", init_params)

    clients = [
        FedClient(i, client_train_loaders[i], device, args.lr, init_params)
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

            # Run local updates on each client
            client_losses = []
            for client in clients:
                local_state, avg_loss = client.local_update(local_steps_per_round)
                client_weights.append(local_state)
                client_losses.append(avg_loss)

            # FedAvg aggregation
            with torch.no_grad():
                avg_state = {}
                for key in global_model.state_dict().keys():
                    avg_state[key] = torch.stack(
                        [w[key].float() for w in client_weights]
                    ).mean(0)
                global_model.load_state_dict(avg_state)

            # Update progress bar with training perplexity
            train_ppl = np.exp(np.mean(client_losses))
            round_pbar.set_postfix({"train_ppl": f"{train_ppl:.2f}"})

            # Evaluate and save params/deltas at specified frequency
            if global_step % args.val_freq == 0 or global_step == 1:
                val_loss = evaluate(global_model, global_val_loader, device)
                val_ppl = np.exp(val_loss)

                # Server: save params + deltas
                curr_params = get_ssm_params(global_model)
                save_npz(
                    f"{exp_dir}/server/params/step_{global_step:06d}.npz", curr_params
                )
                delta_server = {
                    f"delta_{k}": curr_params[k] - init_params[k] for k in init_params
                }
                save_npz(
                    f"{exp_dir}/server/deltas/step_{global_step:06d}.npz", delta_server
                )

                # Clients: save params + deltas
                client_val_ppls = []
                for i, client in enumerate(clients):
                    client.model.eval()
                    client_val_loss = evaluate(client.model, global_val_loader, device)
                    client_val_ppls.append(np.exp(client_val_loss))

                    curr_client_params = get_ssm_params(client.model)
                    save_npz(
                        f"{exp_dir}/client_{i}/params/step_{global_step:06d}.npz",
                        curr_client_params,
                    )
                    delta_client = {
                        f"delta_{k}": curr_client_params[k] - init_params[k]
                        for k in init_params
                    }
                    save_npz(
                        f"{exp_dir}/client_{i}/deltas/step_{global_step:06d}.npz",
                        delta_client,
                    )

                # Generate text
                gen_text = generate_text(global_model, tokenizer, device=device)

                record = {
                    "global_step": global_step,
                    "train_ppl": float(train_ppl),
                    "server_val_ppl": float(val_ppl),
                    "client_val_ppls": [float(ppl) for ppl in client_val_ppls],
                    "generated": gen_text,
                }
                history.append(record)

                logging.info(
                    f"Step {global_step:4d} | Train PPL: {train_ppl:.2f} | "
                    f"Server Val PPL: {val_ppl:.2f} | Client Val PPLs: {[f'{p:.2f}' for p in client_val_ppls]}"
                )

    save_json(history, f"{exp_dir}/training_history.json")
    logging.info(f"Federated training complete. Results saved to {exp_dir}")

    # Future: Add regularization