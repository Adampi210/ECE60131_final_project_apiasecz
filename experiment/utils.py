import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import ssm_config

##### SETUP & IO #####
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

def save_npz(filepath, data_dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, **{k: v.numpy() for k, v in data_dict.items()})

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_ssm_params(model):
    """
    Extract Mamba2/SSM specific parameters.
    Args:
        model: The model from which to extract parameters.
    Returns:
        A dictionary of extracted parameters.
    """
    params = {}
    for name, param in model.named_parameters():
        if "mixer" not in name:
            continue
        # Extract only relevant SSM parameters
        if any(key in name for key in ["in_proj.weight", "A_log", "D", "dt_bias"]):
            clean_name = (
                name.replace(".weight", "").replace(".bias", "").split("mixer.")[-1]
            )
            params[clean_name] = param.detach().cpu().clone()
    return params

##### DATA LOADING #####
def collate_fn(batch):
    """
    Custom collate function to stack input_ids and labels.
    Args:
        batch: A list of dataset items.
    Returns:
        A dictionary with stacked input_ids and labels.
    NOTE: Training SSMs on WikiText2 requires a custom collate function
    that stacks input_ids and labels properly. If not specified, the 
    default collate function will not work as expected.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}

##### MODEL & EVALUATION #####
def init_mamba(config_name, vocab_size):
    """
    Initialize Mamba model with specified configuration and vocabulary size.
    Args:
        config_name: Name of the configuration to use.
        vocab_size: Vocabulary size for the model.
    Returns:
        An instance of MambaLMHeadModel.
    """
    config = ssm_config.configs[config_name]
    config.vocab_size = vocab_size
    return MambaLMHeadModel(config)

def generate_text(model, tokenizer, prompt="Once upon a time,", max_length=100, device="cpu"):
    """
    Generate sample text using the model given a prompt.
    Args:
        model: The language model to use for generation.
        tokenizer: The tokenizer corresponding to the model.
        prompt: The initial text prompt to start generation.
        max_length: Maximum length of the generated text.
        device: Device to run the model on.
    Returns:
        Generated text as a string.
    """
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = ids
    for _ in range(max_length):
        with torch.no_grad():
            next_token = model(generated).logits[:, -1:].argmax(dim=-1)
        generated = torch.cat([generated, next_token], dim=1)
    model.train()
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def evaluate(model, val_loader, device, criterion=nn.CrossEntropyLoss()):
    """
    Evaluate the model on the validation dataset.
    Args:
        model: The language model to evaluate.
        val_loader: DataLoader for the validation dataset.
        device: Device to run the model on.
        criterion: Loss function to use for evaluation.
    Returns:
        Average loss over the validation dataset.
    """
    model.eval()
    total_loss = 0.0
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
