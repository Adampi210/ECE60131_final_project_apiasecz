import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

##### Helper Functions #####
def get_step_from_filename(filepath):
    """
    Extract step number from 'step_000100.npz' filename.
    Args:
        filepath: Full path to the file.
    """
    basename = os.path.basename(filepath)
    return int(basename.split("_")[1].split(".")[0])

def find_seed_directories(data_dir):
    """
    Find all directories matching 'central_*_seed_*'.
    Args:
        data_dir: Root directory to search within.
    """
    pattern = os.path.join(data_dir, "central_*_seed_*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print(f"No seed directories found in {data_dir} matching 'central_seed_*'")
    return dirs

def get_delta_matrix(data, target_layer="layer_0"):
    """
    Extract the weight delta for a specific layer.
    Default to layer_0 to compare consistently across depths.
    Args:
        data: Loaded npz data containing deltas.
        target_layer: Layer identifier string.
    """
    # Try exact match for the specific layer's projection
    # Key usually looks like: 'delta_layers.0.mixer.in_proj.weight' or 'delta_layer_0_in_proj'

    # Heuristic: search keys containing "in_proj" and the layer index
    keys = [k for k in data.keys() if "in_proj" in k]

    # Filter for specific layer if possible (e.g. 'layer_0' or 'layers.0')
    layer_keys = [
        k
        for k in keys
        if f"{target_layer}" in k or f"layers.{target_layer.split('_')[-1]}" in k
    ]

    # If found specific layer keys, use the first one
    if layer_keys:
        return data[layer_keys[0]]

    # Fallback: If no layer index found (e.g. 1-layer model might not index), return first available
    if keys:
        return data[keys[0]]

    return None

##### Plotting Functions #####
def plot_heatmaps_per_seed(seed_dir, output_dir):
    """
    Generate the 'Low-Rank Adaptation' heatmap grid for a specific seed.
    Sort by L2 norm to reveal structure.
    Args:
        seed_dir: Directory containing deltas for a specific seed.
        output_dir: Where to save the generated heatmap.
    """
    seed_name = os.path.basename(seed_dir)
    delta_dir = os.path.join(seed_dir, "deltas")

    files = sorted(
        glob.glob(os.path.join(delta_dir, "step_*.npz")), key=get_step_from_filename
    )

    if not files:
        return
    # Select a subset of steps (e.g., 10 evenly spaced)
    if len(files) > 10:
        indices = np.linspace(0, len(files) - 1, 10, dtype=int)
        selected_files = [files[i] for i in indices]
    else:
        selected_files = files

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    im = None

    # Analyze Layer 0 for heatmaps
    target_layer = "layer_0"

    for idx, path in enumerate(selected_files):
        try:
            data = np.load(path)
            delta = get_delta_matrix(data, target_layer)

            if delta is None:
                continue

            # Sort rows/cols by L2 norm
            row_norms = np.linalg.norm(delta, axis=1)
            col_norms = np.linalg.norm(delta, axis=0)
            row_order = np.argsort(-row_norms)
            col_order = np.argsort(-col_norms)

            top_k = min(80, delta.shape[0], delta.shape[1])
            zoomed = delta[row_order[:top_k]][:, col_order[:top_k]]

            im = axes[idx].imshow(
                zoomed, cmap="RdBu_r", aspect="auto", vmin=-0.06, vmax=0.06
            )

            step = get_step_from_filename(path)
            axes[idx].set_title(f"Step {step}", fontsize=12)
            axes[idx].axis("off")

        except Exception as e:
            print(f"Error processing {path}: {e}")

    plt.suptitle(
        f"Layer Structure ({target_layer}) - {seed_name}",
        fontsize=16,
    )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Δ Magnitude")

    save_path = os.path.join(output_dir, f"heatmap_structure_{seed_name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved heatmap for {seed_name}")

def aggregate_svd_data(seed_dirs, config_name):
    """
    Compute SVD for all steps across all seeds for a specific config.
    Args:
        seed_dirs: List of directories for each seed.
        config_name: Name of the configuration.
    Returns:
        steps: Sorted list of unique training steps.
        svd_history: Dict mapping step -> list of SVD singular value arrays.
    """
    svd_history = defaultdict(list)
    all_steps = set()

    # Use first layer for consistency
    target_layer = "layer_0"

    print(f"Computing SVDs for config: {config_name} (Layer: {target_layer})...")

    for s_dir in seed_dirs:
        delta_dir = os.path.join(s_dir, "deltas")
        files = glob.glob(os.path.join(delta_dir, "step_*.npz"))

        for f in files:
            step = get_step_from_filename(f)
            all_steps.add(step)

            try:
                data = np.load(f)
                delta = get_delta_matrix(data, target_layer)

                if delta is not None:
                    S = np.linalg.svd(delta, full_matrices=False, compute_uv=False)
                    svd_history[step].append(S)

            except Exception as e:
                print(f"Error SVD in {f}: {e}")

    return sorted(list(all_steps)), svd_history

def plot_averaged_svd_evolution(steps, svd_history, output_dir, config_name):
    """
    Plot the evolution of averaged singular values over training steps.
    Args:
        steps: Sorted list of training steps.
        svd_history: Dict mapping step -> list of SVD singular value arrays.
        output_dir: Where to save the plot.
        config_name: Name of the configuration.
    """
    
    if not steps:
        return

    means = []
    stds = []
    valid_steps = []

    for step in steps:
        s_list = svd_history[step]
        if not s_list:
            continue
        try:
            stack = np.stack(s_list)
            means.append(np.mean(stack, axis=0))
            stds.append(np.std(stack, axis=0))
            valid_steps.append(step)
        except ValueError:
            continue

    means = np.array(means)
    stds = np.array(stds)

    if len(means) == 0:
        return

    plt.figure(figsize=(12, 7))

    num_sv_to_plot = min(8, means.shape[1])
    colors = plt.cm.tab10(np.linspace(0, 1, num_sv_to_plot))

    for i in range(num_sv_to_plot):
        m = means[:, i]
        s = stds[:, i]
        plt.plot(valid_steps, m, label=f"σ{i+1}", color=colors[i], linewidth=2)
        plt.fill_between(valid_steps, m - s, m + s, color=colors[i], alpha=0.2)

    plt.yscale("log")
    plt.xlabel("Training Step")
    plt.ylabel("Singular Value (log scale)")
    plt.title(f"SVD Evolution: {config_name}\n(Layer 0, Mean ± Std)")
    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="-", alpha=0.2)

    save_path = os.path.join(output_dir, f"averaged_svd_evolution_{config_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_averaged_final_spectrum(steps, svd_history, output_dir, config_name):
    """
    Plot the final averaged singular value spectrum with 95% energy rank.
    Args:
        steps: Sorted list of training steps.
        svd_history: Dict mapping step -> list of SVD singular value arrays.
        output_dir: Where to save the plot.
        config_name: Name of the configuration.
    """
    if not steps:
        return

    final_step = steps[-1]
    s_list = svd_history[final_step]

    if not s_list:
        return

    stack = np.stack(s_list)
    mean_spectrum = np.mean(stack, axis=0)
    std_spectrum = np.std(stack, axis=0)

    squared_S = mean_spectrum**2
    total_energy = np.sum(squared_S)
    cum_energy = np.cumsum(squared_S) / total_energy

    rank_95 = np.argmax(cum_energy >= 0.95) + 1

    plt.figure(figsize=(10, 6))
    x_indices = np.arange(len(mean_spectrum))

    plt.plot(
        x_indices, mean_spectrum, label="Mean Spectrum", color="tab:blue", linewidth=2
    )
    plt.fill_between(
        x_indices,
        mean_spectrum - std_spectrum,
        mean_spectrum + std_spectrum,
        color="tab:blue",
        alpha=0.3,
    )

    plt.axvline(
        rank_95, color="orange", linestyle="--", label=f"95% Energy (Rank {rank_95})"
    )
    
    plt.yscale("log")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Magnitude (log scale)")
    plt.title(f"Final Spectrum: {config_name} (Step {final_step})")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    save_path = os.path.join(output_dir, f"averaged_final_spectrum_{config_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root folder containing 'central_*_seed_*'",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to save plots"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find Seeds
    seed_dirs = find_seed_directories(args.data_dir)
    if not seed_dirs:
        exit()

    # Per-Seed Heatmaps
    print("\n--- Generating Per-Seed Heatmaps ---")
    for s_dir in seed_dirs:
        plot_heatmaps_per_seed(s_dir, args.output_dir)

    # Group by configuration
    config_groups = defaultdict(list)
    for s_dir in seed_dirs:
        dirname = os.path.basename(s_dir)
        try:
            prefix_part = dirname.split("_seed_")[0]
            config_name = prefix_part.replace("central_config_", "")
            config_groups[config_name].append(s_dir)
        except:
            print(f"Skipping malformed directory name: {dirname}")

    # Generate SVD Plots per config
    print("\n--- Computing Averaged Dynamics Per Configuration ---")
    for config_name, dirs in config_groups.items():
        print(f"Processing config: {config_name} ({len(dirs)} seeds)")
        steps, svd_hist = aggregate_svd_data(dirs, config_name)
        plot_averaged_svd_evolution(steps, svd_hist, args.output_dir, config_name)
        plot_averaged_final_spectrum(steps, svd_hist, args.output_dir, config_name)

    print("\nDone.")
