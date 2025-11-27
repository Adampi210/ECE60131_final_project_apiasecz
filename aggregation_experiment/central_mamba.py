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
from einops import rearrange
import ssm_config

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Gets the available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_fn(batch):
    """Custom collate function to handle the sequences in the dataset."""
    return {
        'input_ids': torch.stack([torch.tensor(example['input_ids']) for example in batch]),
        'attention_mask': torch.stack([torch.tensor(example['attention_mask']) for example in batch]),
        'labels': torch.stack([torch.tensor(example['labels']) for example in batch]),
    }

def get_dataloader(split, sequence_length, batch_size, cache_dir, tokenizer):
    """Loads and processes the WikiText-2 dataset"""
    # Load data from cache
    dataset_cache = os.path.join(cache_dir, f"wikitext-2-processed-{split}-seq{sequence_length}")
    try:
        logging.info(f"Attempting to load processed dataset from disk: {dataset_cache}")
        dataset = load_from_disk(dataset_cache)
        logging.info("Successfully loaded processed dataset from disk.")
    # If not found, create the processed dataset
    except FileNotFoundError:
        logging.info("Processed dataset not found on disk. Starting raw data preparation.")
        try:
            os.environ['HF_DATASETS_OFFLINE'] = "1"
            logging.info("Attempting to load RAW dataset from cache in OFFLINE mode.")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)
            logging.info("Successfully loaded raw dataset from cache.")
        except Exception as e:
            logging.warning(f"Raw dataset not found in cache ({e}). Switching to online mode to download.")
            del os.environ['HF_DATASETS_OFFLINE']
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)
            logging.info("Successfully downloaded raw dataset.")
        finally:
            if 'HF_DATASETS_OFFLINE' in os.environ:
                del os.environ['HF_DATASETS_OFFLINE']
        
        def tokenize_function(examples):
            return tokenizer(examples["text"])
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"], desc="Tokenizing dataset")
        
        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // sequence_length) * sequence_length
            result = {k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)] for k, t in concatenated_examples.items()}
            result["labels"] = result["input_ids"].copy()
            return result

        dataset = tokenized_dataset.map(group_texts, batched=True, batch_size=1000, num_proc=4, desc=f"Grouping texts into blocks of {sequence_length}")
        
        logging.info(f"Saving processed dataset to {dataset_cache}")
        dataset.save_to_disk(dataset_cache)

    # Create a sampler that randomly samples from the entire dataset for each batch
    # replacement=True ensures we can sample more than the dataset size
    # num_samples=len(dataset) keeps the 'epoch' length consistent with the original dataset size
    sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    return dataloader

def get_WikiText_2(sequence_length, batch_size, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    try:
        logging.info(f"Attempting to load tokenizer 'gpt2' from local cache: {cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir, local_files_only=True)
        logging.info("Successfully loaded tokenizer from local cache.")
    except (OSError, ValueError):
        logging.warning(f"Tokenizer 'gpt2' not found in cache. Downloading and caching to {cache_dir}...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
        logging.info("Successfully downloaded and cached tokenizer.")

    train_dataloader = get_dataloader('train', sequence_length, batch_size, cache_dir, tokenizer)
    val_dataloader = get_dataloader('validation', sequence_length, batch_size, cache_dir, tokenizer)
    return train_dataloader, val_dataloader, tokenizer

def init_mamba(config_name, vocab_size):
    logging.info(f"Initializing Mamba model with config: {config_name}")
    device = get_device()
    config = ssm_config.configs[config_name]
    config.vocab_size = vocab_size
    model = MambaLMHeadModel(config)
    model = model.to(device)
    return model

def generate_text(model, tokenizer, prompt, max_length=100, device='cpu'):
    """Generates text from the model using a fixed prompt."""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
    generated = input_ids
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
    model.train()
    return tokenizer.decode(generated[0])

def train(model, train_dataloader, val_dataloader, tokenizer, n_epochs, lr, device, val_freq=10):
    model.to(device)
    # OPTIM USED: AdamW, weight_decay = 0.01
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    logging.info(f"Starting training for {n_epochs} epochs. Validation every {val_freq} steps.")
    history = []
    global_step = 0
    
    for epoch in range(n_epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            global_step += 1
            inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(inputs).logits[:, :-1, :].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            batch_perplexity = torch.exp(loss).item()
            history.append({"type": "train", "step": global_step, "loss": loss.item(), "perplexity": batch_perplexity})
            pbar.set_postfix({'loss': loss.item(), 'ppl': batch_perplexity})
            if global_step % val_freq == 0:
                avg_val_loss, val_perplexity = evaluate(model, val_dataloader, device, criterion)
                logging.info(f"Step {global_step} | Val Loss: {avg_val_loss:.4f} | Val Perplexity: {val_perplexity:.2f}")
                history.append({"type": "val", "step": global_step, "loss": avg_val_loss, "perplexity": val_perplexity})
                # Generate text to track output evolution
                generated = generate_text(model, tokenizer, "Once upon a time,", max_length=100, device=device)
                history.append({"type": "gen", "step": global_step, "text": generated})
    return history

def evaluate(model, val_dataloader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            logits = model(inputs).logits[:, :-1, :].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    model.train()
    return avg_loss, perplexity

def save_metric_to_json(data, metric_name, directory, seed):
    file_path = os.path.join(directory, f"{metric_name}_seed_{seed}.json")
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved {metric_name} data to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save {metric_name} data: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Train a Mamba model on WikiText-2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default="pure_ssm_1_layer", help="Config name from ssm_config.py")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sequence_length', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--data_dir', type=str, default='../../data/results/', help='Base directory to save results.')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    logging.info(f"Running with seed {args.seed} on device: {device}")
    
    lr_str = str(args.lr).replace("0.", "")
    dir_identifier = (
        f"centralized_config_{args.config}_"
        f"bs_{args.batch_size}_seq_{args.sequence_length}_"
        f"ep_{args.n_epochs}_lr_{lr_str}"
    )
    
    experiment_dir = os.path.join(args.data_dir, dir_identifier)
    os.makedirs(experiment_dir, exist_ok=True)
    logging.info(f"Results will be saved in: {experiment_dir}")

    logging.info("Preparing data...")
    cache_dir = "/scratch/gilbreth/apiasecz/data/cache/"
    train_dataloader, val_dataloader, tokenizer = get_WikiText_2(
        sequence_length=args.sequence_length, batch_size=args.batch_size, cache_dir=cache_dir
    )
    vocab_size = tokenizer.vocab_size

    logging.info("Initializing Model...")
    model = init_mamba(config_name=args.config, vocab_size=vocab_size)
    
    logging.info("Starting Training...")
    training_history = train(model, train_dataloader, val_dataloader, tokenizer, args.n_epochs, args.lr, device, val_freq=10)

    logging.info("Training finished.")
    
    train_losses = [item['loss'] for item in training_history if item['type'] == 'train']
    train_perplexities = [item['perplexity'] for item in training_history if item['type'] == 'train']
    val_losses = [item['loss'] for item in training_history if item['type'] == 'val']
    val_perplexities = [item['perplexity'] for item in training_history if item['type'] == 'val']
    generated_texts = [item for item in training_history if item['type'] == 'gen']

    save_metric_to_json(train_losses, 'train_loss', experiment_dir, args.seed)
    save_metric_to_json(train_perplexities, 'train_perplexity', experiment_dir, args.seed)
    save_metric_to_json(val_losses, 'val_loss', experiment_dir, args.seed)
    save_metric_to_json(val_perplexities, 'val_perplexity', experiment_dir, args.seed)
    save_metric_to_json(generated_texts, 'generated_texts', experiment_dir, args.seed)

    # Save model parameters
    model_path = os.path.join(experiment_dir, f"model_seed_{args.seed}.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Saved model parameters to {model_path}")

    # Sanity check: Generate final text
    final_generated = generate_text(model, tokenizer, "Once upon a time,", max_length=200, device=device)
    logging.info(f"Sanity check - Final generated text: {final_generated}")