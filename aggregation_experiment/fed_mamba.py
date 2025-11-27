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
from collections import deque

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
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }

def get_dataloaders(num_clients, sequence_length, batch_size, cache_dir):
    """Loads and processes the WikiText-2 dataset"""
    os.makedirs(cache_dir, exist_ok=True)
    # Use pretrained tokenizer, GPT-2 
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir, local_files_only=True)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

    def get_dataset(split):
        processed_path = os.path.join(cache_dir, f"wikitext-2-processed-{split}-seq{sequence_length}")
        try:
            return load_from_disk(processed_path)
        except FileNotFoundError:
            try:
                os.environ['HF_DATASETS_OFFLINE'] = "1"
                raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)
            except Exception:
                del os.environ['HF_DATASETS_OFFLINE']
                raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=cache_dir)
            finally:
                if 'HF_DATASETS_OFFLINE' in os.environ: del os.environ['HF_DATASETS_OFFLINE']
            
            tokenized = raw_dataset.map(lambda e: tokenizer(e["text"]), batched=True, num_proc=4, remove_columns=["text"])
            
            def group_texts(examples):
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                total_length = (total_length // sequence_length) * sequence_length
                result = {k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)] for k, t in concatenated_examples.items()}
                result["labels"] = result["input_ids"].copy()
                return result

            grouped = tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=4)
            grouped.save_to_disk(processed_path)
            return grouped

    train_dataset = get_dataset("train")
    val_dataset = get_dataset("validation")
    
    # Server (Global) Validation Loader
    val_dataset.set_format("torch")
    val_sampler = RandomSampler(val_dataset, replacement=True, num_samples=len(val_dataset))
    server_val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn)
    
    # Split Training Data for Clients
    train_samples_per_client = len(train_dataset) // num_clients
    client_train_dataloaders = []
    for i in range(num_clients):
        start, end = i * train_samples_per_client, (i + 1) * train_samples_per_client
        client_ds = train_dataset.select(range(start, end))
        client_ds.set_format("torch")
        client_sampler = RandomSampler(client_ds, replacement=True, num_samples=len(client_ds)*100) # Sample much more to avoid exhaustion
        client_train_dataloaders.append(DataLoader(client_ds, batch_size=batch_size, sampler=client_sampler, collate_fn=collate_fn))

    # Split Validation Data for Clients
    val_samples_per_client = len(val_dataset) // num_clients
    client_val_dataloaders = []
    for i in range(num_clients):
        start, end = i * val_samples_per_client, (i + 1) * val_samples_per_client
        # Ensure we don't go out of bounds if division isn't perfect, though select handles ranges well
        if start >= len(val_dataset): break 
        
        client_val_ds = val_dataset.select(range(start, end))
        client_val_ds.set_format("torch")
        # Standard sampler for validation (no replacement usually needed, but keeping consistent or simple)
        client_val_sampler = RandomSampler(client_val_ds, replacement=False) 
        client_val_dataloaders.append(DataLoader(client_val_ds, batch_size=batch_size, sampler=client_val_sampler, collate_fn=collate_fn))
    
    return client_train_dataloaders, client_val_dataloaders, server_val_dataloader, tokenizer

############## MODEL & FL COMPONENTS ##############
def init_mamba(config_name, vocab_size):
    logging.info(f"Initializing Mamba model with config: {config_name}")
    config = ssm_config.configs[config_name]
    config.vocab_size = vocab_size  # Override vocab_size
    model = MambaLMHeadModel(config)
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

class Client:
    def __init__(self, client_id, train_dataloader, val_dataloader, device, lr, config_name, vocab_size):
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.model = init_mamba(config_name, vocab_size).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
    
    def local_train(self, dataloader_iter, num_updates):
        self.model.train()
        losses, perplexities = [], []
        for _ in range(num_updates):
            try:
                batch = next(dataloader_iter)
                inputs, targets = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(inputs).logits[:, :-1, :].contiguous()
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                perplexities.append(torch.exp(loss).item())
            except StopIteration:
                logging.warning(f"Client {self.client_id} dataloader exhausted during local training.")
                break
        return self.model.state_dict(), losses, perplexities

    def evaluate(self):
        """Evaluates the model on the client's local validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                logits = self.model(inputs).logits[:, :-1, :].contiguous()
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item() if num_batches > 0 else 0.0
        self.model.train()
        return avg_loss, perplexity

class Server:
    def __init__(self, device, config_name, vocab_size):
        self.device = device
        self.model = init_mamba(config_name, vocab_size).to(device)
        self.criterion = nn.CrossEntropyLoss()

    def aggregate_weights(self, client_weights):
        with torch.no_grad():
            avg_weights = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.model.state_dict().items()}
            for w in client_weights:
                for k, v in w.items(): avg_weights[k] += v
            for k in avg_weights: avg_weights[k] /= len(client_weights)
            self.model.load_state_dict(avg_weights)

    def evaluate(self, val_dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, targets = batch['input_ids'].to(self.device), batch['labels'].to(self.device)
                logits = self.model(inputs).logits[:, :-1, :].contiguous()
                loss = self.criterion(logits.view(-1, logits.size(-1)), targets[:, 1:].contiguous().view(-1))
                total_loss += loss.item()
        avg_loss = total_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        self.model.train()
        return avg_loss, perplexity

def get_ssm_tensors(model):
    """Extracts SSM-related parameter tensors from the model."""
    tensors = {}
    for l_idx, layer in enumerate(model.backbone.layers):
        mixer = layer.mixer
        prefix = f"layer_{l_idx}_"
        tensors[prefix + "A_log"] = mixer.A_log.detach().cpu()
        tensors[prefix + "in_proj.weight"] = mixer.in_proj.weight.detach().cpu()
        tensors[prefix + "dt_bias"] = mixer.dt_bias.detach().cpu()
        tensors[prefix + "D"] = mixer.D.detach().cpu()
    return tensors

def get_ssm_stats(tensors):
    """Computes statistics (mean, variance, L2 norm) for each tensor."""
    stats = {}
    for k, v in tensors.items():
        stats[k] = {
            'mean': v.mean().item(),
            'var': v.var().item(),
            'l2_norm': torch.norm(v).item()
        }
    return stats

def save_metric_to_json(data, metric_name, directory, seed):
    grouped_data = {}
    
    # Handle Training Metrics (already grouped by client ID in history)
    if metric_name in ['train_loss', 'train_perplexity']:
        for entry in data:
            key = f"client_{entry['client']}"
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(entry['loss'] if metric_name == 'train_loss' else entry['perplexity'])
            
    # Handle Validation Metrics (Server + individual Clients)
    elif metric_name in ['val_loss', 'val_perplexity']:
        for entry in data:
            client_id = entry.get('client', 'server')
            # Normalize key name
            if client_id == 'server':
                key = 'server'
            else:
                key = f"client_{client_id}"
                
            if key not in grouped_data:
                grouped_data[key] = []
            
            # Append metric
            metric_val = entry['loss'] if metric_name == 'val_loss' else entry['perplexity']
            grouped_data[key].append(metric_val)

    elif metric_name == 'generated_texts':
        for entry in data:
            key = entry.get('client', 'server')  # Use 'client' if present, else 'server'
            if isinstance(key, int):
                key = f"client_{key}"
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(entry['text'])
    
    path = os.path.join(directory, f"{metric_name}_seed_{seed}.json")
    with open(path, 'w') as f: json.dump(grouped_data, f, indent=4)
    logging.info(f"Saved {metric_name} to {path}")

def save_tensors_to_npz(filepath, tensor_dict):
    npz_dict = {k: v.numpy() for k, v in tensor_dict.items()}
    np.savez(filepath, **npz_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Per-Batch Synchronous Federated Learning for Mamba")
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--local_updates', type=int, default=2, help='Number of local gradient descent steps per client per round.')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sequence_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--config", type=str, default="pure_ssm_1_layer", help="Config name from ssm_config.py")
    parser.add_argument('--data_dir', type=str, default="../../data/results")
    parser.add_argument('--cache_dir', type=str, default="/scratch/gilbreth/apiasecz/data/cache/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = get_device()
    set_seed(args.seed)

    dir_identifier = f"federated_nclients_{args.num_clients}_config_{args.config}_bs_{args.batch_size}_seq_{args.sequence_length}_ep_{args.n_epochs}_lu_{args.local_updates}_lr_{str(args.lr).replace('0.', '')}"
    experiment_dir = os.path.join(args.data_dir, dir_identifier)
    os.makedirs(experiment_dir, exist_ok=True)
    logging.info(f"Results will be saved in: {experiment_dir}")

    # Updated to return val dataloaders for clients
    client_train_loaders, client_val_loaders, server_val_dataloader, tokenizer = get_dataloaders(args.num_clients, args.sequence_length, args.batch_size, args.cache_dir)
    vocab_size = tokenizer.vocab_size
    
    server = Server(device, args.config, vocab_size)
    # Initialize clients with both train and val loaders
    clients = [Client(i, client_train_loaders[i], client_val_loaders[i], device, args.lr, args.config, vocab_size) for i in range(args.num_clients)]

    # Initialize parameter histories
    server_param_history = {
        'tensors_init': get_ssm_tensors(server.model),
        'tensor_deque': deque(maxlen=10),
        'stats': []
    }
    client_param_histories = []
    for i in range(args.num_clients):
        client_param_histories.append({
            'tensors_init': get_ssm_tensors(clients[i].model),
            'tensor_deque': deque(maxlen=10),
            'stats': []
        })

    history = []
    comm_round = 0
    
    # Calculate total communication rounds
    steps_per_epoch = min(len(dl.sampler.data_source) for dl in client_train_loaders) // args.batch_size
    total_comm_rounds = (steps_per_epoch // args.local_updates) * args.n_epochs
    
    logging.info(f"Total communication rounds to be performed: {total_comm_rounds}")

    for epoch in range(args.n_epochs):
        dataloader_iters = [iter(c.train_dataloader) for c in clients]
        
        # The number of rounds per epoch depends on local_updates
        rounds_per_epoch = steps_per_epoch // args.local_updates
        pbar = tqdm(range(rounds_per_epoch), desc=f"Epoch {epoch+1}/{args.n_epochs} (Comm. Rounds)")
        
        for step in pbar:
            comm_round += 1
            
            global_weights = server.model.state_dict()
            client_updates, round_losses = [], []

            for idx, client in enumerate(clients):
                client.model.load_state_dict(global_weights)
                local_weights, losses, perplexities = client.local_train(dataloader_iters[client.client_id], args.local_updates)
                
                # Collect SSM tensors and stats for client after local training
                current_tensors = get_ssm_tensors(client.model)
                client_param_histories[idx]['tensor_deque'].append(current_tensors)
                stats_dict = get_ssm_stats(current_tensors)
                client_param_histories[idx]['stats'].append({
                    'comm_round': comm_round,
                    **stats_dict
                })
                
                client_updates.append(local_weights)
                round_losses.extend(losses)

                for i in range(len(losses)):
                    history.append({
                        "type": "train", "comm_round": comm_round, "client": client.client_id, 
                        "loss": losses[i], "perplexity": perplexities[i]
                    })
            
            if round_losses:
                pbar.set_postfix({"avg_loss": f"{np.mean(round_losses):.4f}"})
            
            server.aggregate_weights(client_updates)
            
            # Collect SSM tensors and stats for server after aggregation
            current_tensors = get_ssm_tensors(server.model)
            server_param_history['tensor_deque'].append(current_tensors)
            stats_dict = get_ssm_stats(current_tensors)
            server_param_history['stats'].append({
                'comm_round': comm_round,
                **stats_dict
            })

            if comm_round % 10 == 0:
                # Evaluate on Server Global Validation Set
                val_loss, val_perplexity = server.evaluate(server_val_dataloader)
                logging.info(f"Round {comm_round} | Server Val PPL: {val_perplexity:.2f}")
                history.append({
                    "type": "val", "comm_round": comm_round, "client": "server", "loss": val_loss, "perplexity": val_perplexity
                })
                
                # Evaluate on Each Client's Validation Set
                for client in clients:
                    client_val_loss, client_val_ppl = client.evaluate()
                    history.append({
                        "type": "val", "comm_round": comm_round, "client": client.client_id, "loss": client_val_loss, "perplexity": client_val_ppl
                    })

                # Generate text to track output evolution
                generated = generate_text(server.model, tokenizer, "Once upon a time,", max_length=100, device=device)
                history.append({"type": "gen", "comm_round": comm_round, "text": generated})

    logging.info("Federated training finished.")
    
    # Sanity checks
    final_generated_server = generate_text(server.model, tokenizer, "Once upon a time,", max_length=200, device=device)
    logging.info(f"Server - Sanity check - Final generated text: {final_generated_server}")
    history.append({"type": "gen", "comm_round": "final", "text": final_generated_server})
    for client in clients:
        final_generated_client = generate_text(client.model, tokenizer, "Once upon a time,", max_length=200, device=device)
        logging.info(f"Client {client.client_id} - Sanity check - Final generated text: {final_generated_client}")
        history.append({"type": "gen", "comm_round": "final", "client": client.client_id, "text": final_generated_client})
    
    # Save final metrics
    train_entries = [h for h in history if h['type'] == 'train']
    save_metric_to_json(train_entries, 'train_loss', experiment_dir, args.seed)
    save_metric_to_json(train_entries, 'train_perplexity', experiment_dir, args.seed)
    val_entries = [h for h in history if h['type'] == 'val']
    save_metric_to_json(val_entries, 'val_loss', experiment_dir, args.seed)
    save_metric_to_json(val_entries, 'val_perplexity', experiment_dir, args.seed)
    gen_entries = [h for h in history if h['type'] == 'gen']
    save_metric_to_json(gen_entries, 'generated_texts', experiment_dir, args.seed)
    
    # Save SSM parameters and stats
    entities = [('server', server_param_history)] + [(f'client_{i}', client_param_histories[i]) for i in range(args.num_clients)]
    for entity_name, param_history in entities:
        entity_dir = os.path.join(experiment_dir, entity_name)
        os.makedirs(entity_dir, exist_ok=True)
        
        # Save init tensors
        save_tensors_to_npz(os.path.join(entity_dir, f'init_seed_{args.seed}.npz'), param_history['tensors_init'])
        
        # Save stats
        stats_path = os.path.join(entity_dir, f'ssm_stats_seed_{args.seed}.json')
        with open(stats_path, 'w') as f:
            json.dump(param_history['stats'], f, indent=4)
        logging.info(f"Saved SSM stats for {entity_name} to {stats_path}")
        
        if len(param_history['tensor_deque']) > 0:
            # Save final tensors
            final_tensors = list(param_history['tensor_deque'])[-1]
            save_tensors_to_npz(os.path.join(entity_dir, f'final_seed_{args.seed}.npz'), final_tensors)
            
            # Compute and save average of last 10 (or fewer) tensors
            last_tensors = list(param_history['tensor_deque'])
            avg_dict = {}
            for k in last_tensors[0].keys():
                stacked = torch.stack([d[k] for d in last_tensors], dim=0)
                avg_dict[k] = stacked.mean(dim=0)
            save_tensors_to_npz(os.path.join(entity_dir, f'last10_avg_seed_{args.seed}.npz'), avg_dict)
    # Relative change for one number for all matrices
    # TODO: Summarize before every meeting high level summary of every finding from the experiments each time
    # Say a bit outside of the results -> bigger picture 
    