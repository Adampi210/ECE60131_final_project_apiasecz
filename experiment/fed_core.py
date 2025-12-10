import os
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from utils import (
    collate_fn,
)

##### DATA LOADING #####
def get_dataloaders(num_clients, sequence_length, batch_size, cache_dir):
    """
    Load and preprocess the dataset, returning dataloaders 
    for each client and a global validation loader.
    
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", cache_dir=cache_dir, local_files_only=True
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

    # Get the dataset for a given split (train/validation)
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

            # Group texts into sequences of fixed length
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

    # Load datasets
    train_ds = get_dataset("train")
    val_ds = get_dataset("validation")

    # Split train data across clients
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
                subset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        )

    # Global validation loader (full val set)
    val_ds.set_format("torch")
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return client_train_loaders, val_loader, tokenizer

def get_entropy_dataloaders(num_clients, sequence_length, batch_size, cache_dir):
    """
    Get dataloaders for entropy calculation, 
    modified to include local validation sets for each client.
    Args:
        num_clients: Number of clients.
        sequence_length: Sequence length for tokenization.
        batch_size: Batch size for dataloaders.
        cache_dir: Directory to cache datasets and tokenizer.
    Returns:
        client_train_loaders: List of DataLoader objects for each client's training data.
        client_val_loaders: List of DataLoader objects for each client's validation data.   
        val_loader: DataLoader object for global validation data.
        tokenizer: The tokenizer used for data processing.
    """
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
