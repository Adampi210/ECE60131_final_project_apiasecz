# fed_kalman.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import logging
import argparse
import random
import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import ssm_config


# ==========================================
# 1. HELPERS & SETUP (Standard)
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def get_ssm_params(model):
    params = {}
    for name, param in model.named_parameters():
        if "mixer" not in name:
            continue
        if any(key in name for key in ["in_proj.weight", "A_log", "D", "dt_bias"]):
            clean_name = (
                name.replace(".weight", "").replace(".bias", "").split("mixer.")[-1]
            )
            params[clean_name] = param.detach().cpu().clone()
    return params


def save_npz(filepath, data_dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **{k: v.numpy() for k, v in data_dict.items()})


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_text(
    model, tokenizer, prompt="Once upon a time,", max_length=100, device="cpu"
):
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = ids
    for _ in range(max_length):
        with torch.no_grad():
            next_token = model(generated).logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=1)
    model.train()
    return tokenizer.decode(generated[0], skip_special_tokens=True)


# ==========================================
# 2. DATA LOADING (Standard)
# ==========================================
def get_dataloaders(num_clients, sequence_length, batch_size, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", cache_dir=cache_dir, local_files_only=True
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

    def get_dataset(split):
        path = os.path.join(cache_dir, f"wikitext2_{split}_seq{sequence_length}")
        try:
            return load_from_disk(path)
        except:
            raw = load_dataset(
                "wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir
            )
            tokenized = raw.map(
                lambda x: tokenizer(x["text"]), batched=True, remove_columns=["text"]
            )

            def group(examples):
                concat = {k: sum(examples[k], []) for k in examples.keys()}
                total = len(concat["input_ids"])
                total = (total // sequence_length) * sequence_length
                return {
                    k: [
                        t[i : i + sequence_length]
                        for i in range(0, total, sequence_length)
                    ]
                    for k, t in concat.items()
                }

            grouped = tokenized.map(group, batched=True, batch_size=1000)
            grouped = grouped.add_column("labels", grouped["input_ids"])
            grouped.save_to_disk(path)
            return grouped

    train_ds = get_dataset("train")
    val_ds = get_dataset("validation")

    samples_per_client = len(train_ds) // num_clients
    client_train_loaders = []
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else len(train_ds)
        subset = train_ds.select(range(start, end))
        subset.set_format("torch")
        sampler = RandomSampler(subset, replacement=True, num_samples=len(subset) * 10)
        client_train_loaders.append(
            DataLoader(
                subset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn
            )
        )

    val_ds.set_format("torch")
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return client_train_loaders, val_loader, tokenizer


# ==========================================
# 3. CLIENT (Standard FedAvg Client)
# ==========================================
class Client:
    def __init__(self, cid, train_loader, device, lr, init_params):
        self.cid = cid
        self.train_loader = train_loader
        self.device = device
        self.model = None
        self.optimizer = None
        self.init_params = init_params
        self.dataloader_iter = None
        self.lr = lr

    def reset_model(self, global_state_dict, config):
        if self.model is None:
            config = ssm_config.configs[config]
            config.vocab_size = 50257
            self.model = MambaLMHeadModel(config).to(self.device)
        self.model.load_state_dict(global_state_dict)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )
        self.dataloader_iter = iter(self.train_loader)

    def local_update(self, num_updates):
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


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)
            logits = model(inputs).logits[:, :-1, :].contiguous()
            loss = criterion(
                logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1)
            )
            total_loss += loss.item()
    return total_loss / len(val_loader)


# ==========================================
# 4. KALMAN MOMENTUM AGGREGATION LOGIC
# ==========================================
class KalmanServer:
    def __init__(self, model, momentum=0.9, learning_rate=1.0):
        self.model = model
        self.momentum = momentum
        self.server_lr = learning_rate
        # Initialize velocity buffer for every parameter
        self.velocity = {k: torch.zeros_like(p) for k, p in model.state_dict().items()}

    def step(self, client_weights):
        """
        Applies Server-Side Momentum (Kalman Filter style).
        1. Compute pseudo-gradient: G = (w_old - w_avg_clients)
        2. Update velocity: v_new = beta * v_old + G
        3. Update global: w_new = w_old - server_lr * v_new
        """
        # 1. Compute simple average of clients (Measurement)
        avg_weights = {}
        with torch.no_grad():
            for key in client_weights[0].keys():
                avg_weights[key] = torch.stack(
                    [w[key].float() for w in client_weights]
                ).mean(0)

        # 2. Update Global Model
        current_weights = self.model.state_dict()
        new_state_dict = {}

        for key in current_weights.keys():
            # The "gradient" here is the direction towards the client average
            # pseudo_grad = w_old - w_avg
            # We want to move towards w_avg, so direction is (w_avg - w_old)
            direction = avg_weights[key] - current_weights[key].float()

            # Update velocity (Moving Average of the direction)
            # v_{t+1} = beta * v_t + (1 - beta) * direction
            self.velocity[key] = (
                self.momentum * self.velocity[key] + (1.0 - self.momentum) * direction
            )

            # Update weights
            # w_{t+1} = w_t + server_lr * v_{t+1}
            new_state_dict[key] = (
                current_weights[key].float() + self.server_lr * self.velocity[key]
            )

        self.model.load_state_dict(new_state_dict)


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
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
    # New Kalman Args
    parser.add_argument(
        "--server_momentum", type=float, default=0.9, help="Beta for Kalman velocity"
    )
    parser.add_argument(
        "--server_lr", type=float, default=1.0, help="Step size for server update"
    )
    parser.add_argument("--data_dir", type=str, default="./data/results/")
    parser.add_argument("--cache_dir", type=str, default="../../cache/wikitext2")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    exp_name = (
        f"fed_kalman_config_{args.config}_lu_{args.local_updates}_seed_{args.seed}"
    )
    exp_dir = os.path.join(args.data_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/server/params", exist_ok=True)

    logging.info(f"Experiment: {exp_name} (Method: Kalman Momentum)")

    client_train_loaders, global_val_loader, tokenizer = get_dataloaders(
        args.num_clients, args.sequence_length, args.batch_size, args.cache_dir
    )

    config = ssm_config.configs[args.config]
    config.vocab_size = tokenizer.vocab_size
    global_model = MambaLMHeadModel(config).to(device)
    init_params = get_ssm_params(global_model)

    clients = [
        Client(i, client_train_loaders[i], device, args.lr, init_params)
        for i in range(args.num_clients)
    ]

    # Initialize Kalman Server
    server_optimizer = KalmanServer(
        global_model, momentum=args.server_momentum, learning_rate=args.server_lr
    )

    history = []
    global_step = 0
    local_steps_per_round = args.local_updates

    for epoch in range(args.n_epochs):
        for client in clients:
            client.reset_model(global_model.state_dict(), args.config)

        round_pbar = tqdm(
            range(
                len(client_train_loaders[0]) // args.batch_size // local_steps_per_round
            ),
            desc=f"Epoch {epoch+1}/{args.n_epochs}",
        )

        for round_step in round_pbar:
            global_step += 1
            client_weights = []
            client_losses = []

            for client in clients:
                local_state, avg_loss = client.local_update(local_steps_per_round)
                client_weights.append(local_state)
                client_losses.append(avg_loss)

            # === Kalman Aggregation ===
            server_optimizer.step(client_weights)

            train_ppl = np.exp(np.mean(client_losses))
            round_pbar.set_postfix({"train_ppl": f"{train_ppl:.2f}"})

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
    logging.info(f"Kalman Federated training complete. Results saved to {exp_dir}")
