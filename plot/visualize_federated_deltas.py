import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def get_step_from_filename(filepath):
    """Extract step number from 'step_001234.npz'"""
    basename = os.path.basename(filepath)
    return int(basename.split("_")[1].split(".")[0])


def find_federated_dirs(data_dir):
    """
    Find ALL federated-style experiment directories.
    Matches any folder containing '_seed_' but NOT 'central'.
    """
    all_dirs = sorted(glob.glob(os.path.join(data_dir, "*_seed_*")))
    fed_dirs = [d for d in all_dirs if "central" not in os.path.basename(d)]

    if not fed_dirs:
        print(f"No federated directories found in {data_dir}")
    return fed_dirs


def get_group_info(dir_path):
    """
    Extracts (method, config, lu) from directory path.
    Example: 'federated_config_pure_ssm_1_layer_lu_8_seed_0'
    Returns: ('federated', 'pure_ssm_1_layer', '8')
    """
    dirname = os.path.basename(dir_path)

    # 1. Method (everything before _config_)
    if "_config_" not in dirname:
        return "unknown", "unknown", "1"

    parts = dirname.split("_config_")
    method = parts[0]
    rest = parts[1]

    # 2. Config & LU
    # Format is usually: [CONFIG]_lu_[LU]_seed_[SEED]
    # Split by _lu_ to separate config from parameters
    if "_lu_" in rest:
        config_part, params_part = rest.split("_lu_")
        config = config_part
        # Extract LU value (params_part starts with "8_seed_...")
        lu = params_part.split("_seed_")[0]
    else:
        # Fallback if no LU specified (assume 1)
        config = rest.split("_seed_")[0]
        lu = "1"

    return method, config, lu


def get_seed_from_dir(dir_path):
    """Extracts seed number from directory name."""
    dirname = os.path.basename(dir_path)
    try:
        return dirname.split("_seed_")[1].split("_")[0]
    except:
        return "X"


def get_delta_matrix(data):
    """Robustly extracts the main weight matrix (in_proj) from delta dict."""
    key = next((k for k in data.keys() if "in_proj" in k), None)
    if key is None:
        key = next((k for k in data.keys() if "delta" in k and "weight" in k), None)
    if key:
        return data[key]
    return None


def plot_heatmap_for_single_seed(
    exp_dir, output_dir, method, config, lu, entity="server"
):
    """
    Generates ONE heatmap grid for a SPECIFIC seed.
    """
    seed_val = get_seed_from_dir(exp_dir)
    delta_dir = os.path.join(exp_dir, entity, "deltas")

    if not os.path.isdir(delta_dir):
        return

    files = sorted(
        glob.glob(os.path.join(delta_dir, "step_*.npz")),
        key=get_step_from_filename,
    )
    if not files:
        return

    # --- Select up to 8 evenly spaced unique steps ---
    num_plots = min(8, len(files))
    if num_plots < len(files):
        indices = np.linspace(0, len(files) - 1, num_plots, dtype=int)
        indices = sorted(list(set(indices)))
        selected = [files[i] for i in indices]
    else:
        selected = files

    # Setup Grid
    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i in range(len(selected), len(axes)):
        axes[i].axis("off")

    im = None

    for ax, path in zip(axes, selected):
        data = np.load(path)
        delta = get_delta_matrix(data)

        if delta is None:
            ax.set_title("No Param")
            ax.axis("off")
            continue

        # Sort by L2 norm
        row_norms = np.linalg.norm(delta, axis=1)
        col_norms = np.linalg.norm(delta, axis=0)
        row_order = np.argsort(-row_norms)
        col_order = np.argsort(-col_norms)

        top_k = min(80, delta.shape[0], delta.shape[1])
        zoomed = delta[row_order[:top_k]][:, col_order[:top_k]]

        im = ax.imshow(zoomed, cmap="RdBu_r", aspect="auto", vmin=-0.06, vmax=0.06)
        step = get_step_from_filename(path)
        ax.set_title(f"Step {step}", fontsize=11)
        ax.axis("off")

    plt.suptitle(f"{method} | {config} | LU={lu} | Seed {seed_val}", fontsize=16)

    if im:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Δ Magnitude")

    save_path = os.path.join(
        output_dir, f"heatmap_{method}_{config}_lu_{lu}_seed_{seed_val}_{entity}.png"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  -> Saved Heatmap: {os.path.basename(save_path)}")


def aggregate_svd_for_group(exp_dirs, entity="server"):
    """
    Aggregates SVD data across seeds for a (Method, Config, LU) group.
    """
    svd_history = defaultdict(list)
    all_steps = set()

    for exp_dir in exp_dirs:
        delta_dir = os.path.join(exp_dir, entity, "deltas")
        if not os.path.isdir(delta_dir):
            continue

        files = glob.glob(os.path.join(delta_dir, "step_*.npz"))
        for f in files:
            step = get_step_from_filename(f)
            all_steps.add(step)
            try:
                data = np.load(f)
                delta = get_delta_matrix(data)
                if delta is not None:
                    S = np.linalg.svd(delta, full_matrices=False, compute_uv=False)
                    svd_history[step].append(S)
            except:
                pass

    return sorted(list(all_steps)), svd_history


def plot_svd_evolution(steps, svd_history, output_dir, method, config, lu):
    if not steps:
        return

    means, stds, valid_steps = [], [], []
    for step in steps:
        s_list = svd_history[step]
        if not s_list:
            continue
        try:
            stack = np.stack(s_list)
            means.append(np.mean(stack, axis=0))
            stds.append(np.std(stack, axis=0))
            valid_steps.append(step)
        except:
            continue

    if not means:
        return
    means = np.array(means)
    stds = np.array(stds)

    plt.figure(figsize=(10, 6))
    n_plot = min(8, means.shape[1])
    colors = plt.cm.tab10(np.linspace(0, 1, n_plot))

    for i in range(n_plot):
        plt.plot(valid_steps, means[:, i], label=f"σ{i+1}", color=colors[i], lw=2)
        plt.fill_between(
            valid_steps,
            means[:, i] - stds[:, i],
            means[:, i] + stds[:, i],
            color=colors[i],
            alpha=0.2,
        )

    plt.yscale("log")
    plt.xlabel("Training Step")
    plt.ylabel("Singular Value (log)")
    plt.title(f"SVD Evolution: {method} | {config} | LU={lu}")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f"svd_evolution_{method}_{config}_lu_{lu}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_final_spectrum(steps, svd_history, output_dir, method, config, lu):
    if not steps:
        return
    final_step = steps[-1]
    s_list = svd_history[final_step]
    if not s_list:
        return

    stack = np.stack(s_list)
    mean_spec = np.mean(stack, axis=0)
    std_spec = np.std(stack, axis=0)

    sq = mean_spec**2
    cum = np.cumsum(sq) / np.sum(sq)
    r95 = np.argmax(cum >= 0.95) + 1

    plt.figure(figsize=(8, 5))
    x = np.arange(len(mean_spec))
    plt.plot(x, mean_spec, label="Mean Spectrum", color="tab:blue", lw=2)
    plt.fill_between(
        x, mean_spec - std_spec, mean_spec + std_spec, color="tab:blue", alpha=0.3
    )

    plt.axvline(r95, color="orange", ls="--", label=f"95% Energy (Rank {r95})")

    plt.yscale("log")
    plt.title(f"Final Spectrum: {method} | {config} | LU={lu}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(output_dir, f"svd_spectrum_{method}_{config}_lu_{lu}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--entity", type=str, default="server")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_fed_dirs = find_federated_dirs(args.data_dir)

    # Group by (Method, Config, LU)
    groups = defaultdict(list)
    for d in all_fed_dirs:
        method, config, lu = get_group_info(d)
        groups[(method, config, lu)].append(d)

    print(f"Found {len(groups)} unique experiment groups (Method/Config/LU).")

    for (method, config, lu), dirs in groups.items():
        print(f"\nProcessing {method} - {config} - LU={lu} ({len(dirs)} seeds)...")

        # A) Heatmaps: Process EVERY seed individually
        for seed_dir in dirs:
            plot_heatmap_for_single_seed(
                seed_dir, args.output_dir, method, config, lu, args.entity
            )

        # B) SVD Analysis: Aggregate across seeds in this group
        steps, svd_hist = aggregate_svd_for_group(dirs, args.entity)
        plot_svd_evolution(steps, svd_hist, args.output_dir, method, config, lu)
        plot_final_spectrum(steps, svd_hist, args.output_dir, method, config, lu)

    print("\nAll done!")
