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
from collections import deque
from utils import (
    set_seed,
    get_device,
    save_npz,
    save_json,
    get_ssm_params,
    evaluate,
    generate_text,
    init_mamba,
    collate_fn,
)

# Data Loader and Preparation
def get_dataloaders(sequence_length, batch_size, cache_dir):
    """
    Load and preprocess the Wikitext-2 dataset.
    Args:
        sequence_length: The length of each input sequence.
        batch_size: The batch size for the dataloaders.
        cache_dir: Directory to cache the dataset and tokenizer.
    Returns:
        train_dataloader, val_dataloader, tokenizer
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Use gpt2 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", cache_dir=cache_dir, local_files_only=True
        )
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

    # Get dataset for a given split (train/validation)
    def get_dataset(split):
        processed_path = os.path.join(
            cache_dir, f"wikitext-2-processed-{split}-seq{sequence_length}"
        )
        try:
            # Load processed dataset if it exists
            return load_from_disk(processed_path)
        # If not, process and save it
        except FileNotFoundError:
            try:
                os.environ["HF_DATASETS_OFFLINE"] = "1"
                raw_dataset = load_dataset(
                    "wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir
                )
            except Exception:
                del os.environ["HF_DATASETS_OFFLINE"]
                raw_dataset = load_dataset(
                    "wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir
                )
            finally:
                if "HF_DATASETS_OFFLINE" in os.environ:
                    del os.environ["HF_DATASETS_OFFLINE"]

            # Tokenize the dataset
            tokenized = raw_dataset.map(
                lambda e: tokenizer(e["text"]),
                batched=True,
                num_proc=4,
                remove_columns=["text"],
            )

            # Group texts into sequences of fixed length
            def group_texts(examples):
                concatenated_examples = {
                    k: sum(examples[k], []) for k in examples.keys()
                }
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                total_length = (total_length // sequence_length) * sequence_length
                result = {
                    k: [
                        t[i : i + sequence_length]
                        for i in range(0, total_length, sequence_length)
                    ]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            # Group and save the processed dataset
            grouped = tokenized.map(
                group_texts, batched=True, batch_size=1000, num_proc=4
            )
            grouped.save_to_disk(processed_path)
            return grouped

    # Get train and validation datasets
    train_dataset = get_dataset("train")
    val_dataset = get_dataset("validation")

    # Create dataloaders
    # Train dataloader
    train_dataset.set_format("torch")
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=len(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    # Validation dataloader
    val_dataset.set_format("torch")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_dataloader, val_dataloader, tokenizer

# Training
def train(model, train_dl, val_dl, tokenizer, n_epochs, lr, device, exp_dir, val_freq=50):
    # Setup optimizer and loss
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01) # Use AdamW optimizer w/ weight decay
    criterion = nn.CrossEntropyLoss()                                     # Use cross-entropy loss

    # Create directories for saving
    os.makedirs(f"{exp_dir}/deltas", exist_ok=True)
    os.makedirs(f"{exp_dir}/params", exist_ok=True)
    
    history = []
    grad_cosines = []
    prev_grad = None

    # Save initial parameters
    init_params = get_ssm_params(model)
    save_npz(f"{exp_dir}/params/step_000000.npz", init_params)

    global_step = 0
    # Train for n_epochs
    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{n_epochs}")
        # Iterate over batches
        for batch in pbar:
            # Update global step
            global_step += 1

            # Get inputs and targets
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            # Get model outputs, compute loss, backprop, and update parameters
            optimizer.zero_grad()
            logits = model(inputs).logits[:, :-1, :].contiguous()
            loss = criterion(
                logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1)
            )
            loss.backward()
            optimizer.step()

            # Calculate gradient cosine similarity
            grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
            if grads:
                flat = torch.cat(grads)
                if prev_grad is not None:
                    cos = torch.nn.functional.cosine_similarity(
                        prev_grad, flat, dim=0
                    ).item()
                    grad_cosines.append(cos)
                prev_grad = flat.detach()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Evaluate at specified frequency
            if global_step % val_freq == 0:
                model.eval()
                val_loss = evaluate(model, val_dl, device, criterion)
                model.train()

                # Save current parameters
                current_params = get_ssm_params(model)
                save_npz(f"{exp_dir}/params/step_{global_step:06d}.npz", current_params)

                # Compute and save deltas (differences) from initial parameters
                delta_params = {}
                delta_norms = {}
                for key in current_params:
                    delta = current_params[key] - init_params[key]
                    delta_params[f"delta_{key}"] = delta
                    delta_norms[f"norm_delta_{key}"] = torch.norm(delta).item()

                save_npz(f"{exp_dir}/deltas/step_{global_step:06d}.npz", delta_params)

                # Generate sample text
                gen_text = generate_text(
                    model, tokenizer, "Once upon a time,", max_length=100, device=device
                )

                # Log
                record = {
                    "step": global_step,
                    "val_loss": float(val_loss),
                    "val_ppl": float(np.exp(val_loss)),
                    "train_loss": float(loss.item()),
                    "generated": gen_text,
                    "grad_cosine": (
                        float(np.mean(grad_cosines[-10:])) if grad_cosines else None
                    ),
                    "delta_norms": delta_norms,
                }
                history.append(record)

                logging.info(
                    f"Step {global_step} | Val Loss: {val_loss:.4f} | PPL: {np.exp(val_loss):.2f} | in_proj Δnorm: {delta_norms.get('norm_delta_in_proj',0):.6f}"
                )

    return history

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default="pure_ssm_1_layer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--data_dir", type=str, default="../data/results/")
    parser.add_argument("--cache_dir", type=str, default="../../cache/wikitext-2")
    parser.add_argument(
        "--val_freq", type=int, default=10, help="Save deltas every N steps"
    )
    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = get_device()

    # Create experiment directory
    exp_dir = os.path.join(
        args.data_dir, f"central_config_{args.config}_seed_{args.seed}"
    )
    os.makedirs(exp_dir, exist_ok=True)

    # Load data
    logging.info("Loading data...")
    train_dl, val_dl, tokenizer = get_dataloaders(
        args.sequence_length, args.batch_size, args.cache_dir
    )

    # Initialize model
    logging.info("Initializing model...")
    model = init_mamba(args.config, tokenizer.vocab_size)

    # Train model with delta saving
    logging.info("Starting training with delta saving...")
    history = train(
        model,
        train_dl,
        val_dl,
        tokenizer,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        exp_dir=exp_dir,
        val_freq=args.val_freq,
    )

    # Save final history
    save_json(history, os.path.join(exp_dir, "training_history.json"))

    logging.info(f"Training complete. Results saved to {exp_dir}.")
