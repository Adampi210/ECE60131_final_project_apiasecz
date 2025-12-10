# fed_entropy.py
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
# 1. HELPERS & SETUP
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
# 2. DATA LOADING
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
    # We create small local validation sets for entropy calculation
    client_val_loaders = []

    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client if i < num_clients - 1 else len(train_ds)

        # 90/10 split for local train/val
        subset_full = train_ds.select(range(start, end))
        split_idx = int(len(subset_full) * 0.9)
        subset_train = subset_full.select(range(0, split_idx))
        subset_val = subset_full.select(range(split_idx, len(subset_full)))

        subset_train.set_format("torch")
        subset_val.set_format("torch")

        sampler = RandomSampler(
            subset_train, replacement=True, num_samples=len(subset_train) * 10
        )
        client_train_loaders.append(
            DataLoader(
                subset_train,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        )
        client_val_loaders.append(
            DataLoader(
                subset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
            )
        )

    val_ds.set_format("torch")
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return client_train_loaders, client_val_loaders, val_loader, tokenizer


# ==========================================
# 3. CLIENT WITH ENTROPY CALCULATION
# ==========================================
class Client:
    def __init__(self, cid, train_loader, val_loader, device, lr, init_params):
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader  # New: local val loader
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

    def calculate_entropy(self, max_batches=5):
        """Calculates the average entropy of the predictive distribution on validation data."""
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
                # Add epsilon for numerical stability inside log
                batch_entropy = -torch.sum(
                    probs * torch.log(probs + 1e-8), dim=-1
                ).mean()
                entropies.append(batch_entropy.item())

        self.model.train()
        return (
            np.mean(entropies) if entropies else 10.0
        )  # Default high entropy if fails

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

        # Calculate entropy score after training
        entropy_score = self.calculate_entropy()

        return (
            self.model.state_dict(),
            entropy_score,
            np.mean(losses) if losses else 0.0,
        )


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
# 4. ENTROPY AGGREGATION LOGIC
# ==========================================
def entropy_aggregate(global_model, client_weights, client_entropies, temperature=1.0):
    """
    Aggregates weights based on Softmin of entropy.
    alpha_k = exp(-H_k / T) / sum(exp(-H_j / T))
    """
    with torch.no_grad():
        # 1. Calculate weights from entropies
        entropies_tensor = torch.tensor(client_entropies)
        # We use negative entropy because lower entropy is better (higher confidence)
        # Softmax(-H) is equivalent to Softmin(H)
        weights = torch.softmax(-entropies_tensor / temperature, dim=0)

        logging.info(f"Aggregation Weights: {weights.tolist()}")  # Log who got trusted

        avg_state = {}
        first_client_keys = client_weights[0].keys()

        # 2. Weighted Sum
        for key in first_client_keys:
            # Initialize with first client * their weight
            avg_state[key] = client_weights[0][key].float() * weights[0]

            # Add remaining
            for i in range(1, len(client_weights)):
                avg_state[key] += client_weights[i][key].float() * weights[i]

        global_model.load_state_dict(avg_state)


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
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax weighting",
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
        f"fed_entropy_config_{args.config}_lu_{args.local_updates}_seed_{args.seed}"
    )
    exp_dir = os.path.join(args.data_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/server/params", exist_ok=True)

    logging.info(f"Experiment: {exp_name} (Method: Entropy Aggregation)")

    # Note: get_dataloaders now returns local val loaders too
    client_train_loaders, client_val_loaders, global_val_loader, tokenizer = (
        get_dataloaders(
            args.num_clients, args.sequence_length, args.batch_size, args.cache_dir
        )
    )

    config = ssm_config.configs[args.config]
    config.vocab_size = tokenizer.vocab_size
    global_model = MambaLMHeadModel(config).to(device)
    init_params = get_ssm_params(global_model)

    clients = [
        Client(
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
            client_entropies = []
            client_losses = []

            for client in clients:
                local_state, entropy, avg_loss = client.local_update(
                    local_steps_per_round
                )
                client_weights.append(local_state)
                client_entropies.append(entropy)  # Collect entropy
                client_losses.append(avg_loss)

            # === Entropy Aggregation ===
            entropy_aggregate(
                global_model, client_weights, client_entropies, args.temperature
            )

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
                    "avg_client_entropy": float(np.mean(client_entropies)),
                    "generated": gen_text,
                }
                history.append(record)

                logging.info(
                    f"Step {global_step:4d} | PPL: {val_ppl:.2f} | Avg Entropy: {np.mean(client_entropies):.4f}"
                )

    save_json(history, f"{exp_dir}/training_history.json")
    logging.info(f"Entropy Federated training complete. Results saved to {exp_dir}")
