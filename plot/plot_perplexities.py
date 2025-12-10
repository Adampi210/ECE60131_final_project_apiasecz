import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.style.use("seaborn-v0_8-whitegrid")

##### Data Loading and Processing Functions #####
def load_history(path):
    with open(path) as f:
        return json.load(f)

def get_runs(data_dir, method_prefix, config_name, lu=None):
    """
    Finds all 'training_history.json' files matching the criteria.
    Args:
        data_dir: Base directory containing experiment results.
        method_prefix: 'central', 'federated', 'fed_fisher', etc.
        config_name: 'pure_ssm_1_layer'
        lu: integer or None (for central)
    Returns:
        List of loaded histories (one per seed).
    """
    if lu is not None:
        pattern = f"{method_prefix}_config_{config_name}_lu_{lu}_seed_*"
    else:
        pattern = f"{method_prefix}_config_{config_name}_seed_*"
        
    full_pattern = os.path.join(data_dir, pattern, "training_history.json")
    paths = sorted(glob.glob(full_pattern))
    return [load_history(p) for p in paths]

def extract_curves(histories, x_scale=1.0):
    """
    Extracts (steps, train_ppl) and (steps, val_ppl) lists.
    Applies x_scale to steps (useful for LU=8).
    Returns lists of arrays.
    Args:
        histories: List of training histories (dicts).
        x_scale: float, scaling factor for x-axis (steps).
    Returns:
        train_curves: List of (steps, train_ppl) arrays.
        val_curves: List of (steps, val_ppl) arrays.
    """
    train_curves = []
    val_curves = []

    for h in histories:
        t_steps, t_vals = [], []
        v_steps, v_vals = [], []

        for i, rec in enumerate(h):
            # Step handling
            raw_step = rec.get("global_step", rec.get("step", i))
            # Fix: If step is 0 (init), don't scale or scale 0 is 0. 
            # If logic implies step 1 is actually step 8, we multiply.
            step = raw_step * x_scale 

            # Train PPL
            if "train_ppl" in rec:
                t_vals.append(rec["train_ppl"])
                t_steps.append(step)
            elif "train_loss" in rec:
                t_vals.append(np.exp(rec["train_loss"]))
                t_steps.append(step)

            # Val PPL (Prioritize server val, fallback to val_ppl)
            if "server_val_ppl" in rec:
                v_vals.append(rec["server_val_ppl"])
                v_steps.append(step)
            elif "val_ppl" in rec:
                v_vals.append(rec["val_ppl"])
                v_steps.append(step)
            elif "val_loss" in rec:
                v_vals.append(np.exp(rec["val_loss"]))
                v_steps.append(step)

        if t_steps: train_curves.append((np.array(t_steps), np.array(t_vals)))
        if v_steps: val_curves.append((np.array(v_steps), np.array(v_vals)))

    return train_curves, val_curves

def average_curves(curves_list):
    """
    Interpolates curves to a common x-axis and computes Mean/Std.
    Args:
        curves_list: List of (steps, values) arrays.
    Returns:
        common_x: np.array of common steps.
        mean: np.array of mean values at common_x.
        std: np.array of std values at common_x.
    """
    if not curves_list:
        return None, None, None

    # Find max step across all seeds
    max_step = 0
    for steps, _ in curves_list:
        if len(steps) > 0:
            max_step = max(max_step, steps[-1])

    # Define common grid (e.g. every 10 steps)
    # Use the resolution of the first curve as a heuristic
    resolution = 10 
    if len(curves_list[0][0]) > 1:
        resolution = curves_list[0][0][1] - curves_list[0][0][0]

    common_x = np.arange(0, max_step + 1, resolution)
    if common_x[0] == 0 and curves_list[0][0][0] != 0:
        common_x = common_x[1:] # Align start

    ys = []
    for steps, vals in curves_list:
        # Interpolate
        y_interp = np.interp(common_x, steps, vals, left=np.nan, right=np.nan)
        ys.append(y_interp)

    ys = np.array(ys)
    mean = np.nanmean(ys, axis=0)
    std = np.nanstd(ys, axis=0)

    return common_x, mean, std

##### Plotting Functions #####
def plot_central_vs_fed(data_dir, config, out_dir):
    """
    Plot Central vs Federated (LU=1).
    - SCALING: Central steps are scaled to match the final Federated step.
    - STYLE: Validation (Bold, Opaque), Train (Thin, Transparent).
    Args:
        data_dir: Base directory containing experiment results.
        config: Configuration name (e.g. 'pure_ssm_1_layer').
        out_dir: Directory to save plots.
    Returns:
        None
    """
    # Load Data
    cent_runs = get_runs(data_dir, "central", config, lu=None)
    fed_runs = get_runs(data_dir, "federated", config, lu=1)

    if not cent_runs or not fed_runs:
        print(f"Skipping Plot 1: Missing data for {config}")
        return

    c_train, c_val = extract_curves(cent_runs)
    f_train, f_val = extract_curves(fed_runs)

    # Get Averages (Capture X-axis for validation too!)
    cx, ctm, cts = average_curves(c_train)
    cvx, cvm, cvs = average_curves(c_val)

    fx, ftm, fts = average_curves(f_train)
    fvx, fvm, fvs = average_curves(f_val)

    if fx is None or cx is None:
        print("Error: Could not compute curves (empty data?)")
        return

    # Normalize/Scale X-Axis
    max_fed_step = fx[-1]
    max_cent_step = cx[-1]

    if max_cent_step > 0:
        scale_factor = max_fed_step / max_cent_step
        cx = cx * scale_factor
        if cvx is not None:
            cvx = cvx * scale_factor

    plt.figure(figsize=(8, 6))

    # Plot
    col_c = "#1f77b4"
    if cx is not None:
        plt.plot(cx, ctm, color=col_c, lw=1, alpha=0.5, label="Central Train")
        plt.fill_between(cx, ctm - cts, ctm + cts, color=col_c, alpha=0.1)
    if cvx is not None:
        plt.plot(cvx, cvm, color=col_c, lw=2.5, label="Central Val")  # Bold, Opaque

    # Federated (Orange)
    col_f = "#ff7f0e"
    plt.plot(fx, ftm, color=col_f, lw=1, alpha=0.5, label="FedAvg (LU=1) Train")
    plt.fill_between(fx, ftm - fts, ftm + fts, color=col_f, alpha=0.1)
    plt.plot(fvx, fvm, color=col_f, lw=2.5, label="FedAvg (LU=1) Val")  # Bold, Opaque

    plt.yscale("log")
    plt.title(f"Centralized vs Federated (LU=1)\nConfiguration: {config}")
    plt.xlabel(f"Global Steps (Scaled Central x-axis)")
    plt.ylabel("Perplexity (Log Scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"1_central_vs_fed_{config}.png"), dpi=300)
    plt.close()
    print(f"Generated Plot 1 for {config} (Scaled Central x-axis)")

def plot_fed_lu_comparison(data_dir, config, out_dir):
    """
    Compare FedAvg LU=1 vs LU=8.
    Scale LU=8 x-axis by 8.
    Args:
        data_dir: Base directory containing experiment results.
        config: Configuration name (e.g. 'pure_ssm_1_layer').
        out_dir: Directory to save plots.
    Returns:
        None
    """
    runs_1 = get_runs(data_dir, "federated", config, lu=1)
    runs_8 = get_runs(data_dir, "federated", config, lu=8)

    if not runs_1 or not runs_8:
        print(f"Skipping Plot 2: Missing data for {config} (Check if LU=8 exists)")
        return

    # Scale LU=8 by 8
    t1, v1 = extract_curves(runs_1, x_scale=1.0)
    t8, v8 = extract_curves(runs_8, x_scale=8.0) 

    x1, tm1, ts1 = average_curves(t1)
    _, vm1, vs1 = average_curves(v1)
    x8, tm8, ts8 = average_curves(t8)
    _, vm8, vs8 = average_curves(v8)

    min_all = min(min(vm1), min(vm8))
    max_last = max(vm8[-1], vm1[-1])

    plt.figure(figsize=(8, 6))

    # LU 1 (Blue)
    c1 = "tab:blue"
    plt.plot(x1, vm1, color=c1, label="FedAvg LU=1 Val", lw=2)
    plt.fill_between(x1, vm1-vs1, vm1+vs1, color=c1, alpha=0.1)

    # LU 8 (Red)
    c8 = "tab:red"
    plt.plot(x8, vm8, color=c8, label="FedAvg LU=8 Val (Scaled Steps)", lw=2)
    plt.fill_between(x8, vm8-vs8, vm8+vs8, color=c8, alpha=0.1)

    plt.yscale("log")
    plt.ylim(0.95 * min_all, 1.05 * max_last)  # Keep y-limits consistent
    plt.title(f"Impact of Local Updates: LU=1 vs LU=8\nConfig: {config}")
    plt.xlabel("Total Equivalent Steps (Global Steps × LU)")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"2_lu_comparison_{config}.png"), dpi=300)
    plt.close()
    print(f"Generated Plot 2 for {config}")

def plot_method_comparison(data_dir, config, out_dir):
    """
    Compare FedAvg, FedFisher, FedEntropy, FedMomentum (all LU=1).
    - Validation ONLY.
    - Focus on steps > 400 to show convergence differences.
    - Y-axis limits dynamically zoomed to the visible data range.
    Args:
        data_dir: Base directory containing experiment results.
        config: Configuration name (e.g. 'pure_ssm_1_layer').
        out_dir: Directory to save plots.
    Returns:
        None
    """
    methods = {
        "FedAvg": ("federated", "black"),
        "FedFisher": ("fed_fisher", "tab:blue"),
        "FedEntropy": ("fed_entropy", "tab:green"),
        "FedMomentum": ("fed_momentum", "tab:purple"),
    }

    plt.figure(figsize=(10, 7))

    # Track min/max values strictly within the zoomed window (steps > 400)
    zoom_min_y = float("inf")
    zoom_max_y = float("-inf")
    has_data = False

    for label, (prefix, color) in methods.items():
        runs = get_runs(data_dir, prefix, config, lu=1)
        if not runs:
            continue

        # Only validation curves
        _, v_curves = extract_curves(runs)

        xv, vm, vs = average_curves(v_curves)

        if xv is not None:
            # Focus on steps > 400
            mask = xv > 400
            if np.any(mask):
                xv_zoom = xv[mask]
                vm_zoom = vm[mask]
                vs_zoom = vs[mask]

                # Plot Validation
                plt.plot(xv_zoom, vm_zoom, color=color, lw=2.5, label=f"{label}")
                plt.fill_between(
                    xv_zoom,
                    vm_zoom - vs_zoom,
                    vm_zoom + vs_zoom,
                    color=color,
                    alpha=0.15,
                )

                # Update Zoom Limits
                current_min = np.min(vm_zoom - vs_zoom)
                current_max = np.max(vm_zoom + vs_zoom)
                zoom_min_y = min(zoom_min_y, current_min)
                zoom_max_y = max(zoom_max_y, current_max)
                has_data = True

    plt.yscale("log")
    plt.title(
        f"Method Comparison (Validation PPL)\nConfig: {config}, LU=1, Steps > 400"
    )
    plt.xlabel("Global Steps")
    plt.ylabel("Perplexity (Log Scale)")

    # Apply Dynamic Limits if we have data
    if has_data:
        # tight padding around the min/max of the curves
        plt.ylim(zoom_min_y * 0.98, zoom_max_y * 1.02)

    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"3_method_comparison_{config}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated Plot 3 for {config} (Zoomed > 400 steps)")

def plot_multi_config_comparison(data_dir, configs, out_dir):
    """
    Compare multiple configurations for Central vs FedAvg (LU=1).
    - Validation ONLY.
    - Same color for same config.
    - Solid line for Central, Dashed for FedAvg.
    Args:
        data_dir: Base directory containing experiment results.
        configs: List of configuration names (e.g. ['pure_ssm_1_layer', ...]).
        out_dir: Directory to save plots.
    Returns:
        None
    """
    if not configs:
        return

    plt.figure(figsize=(10, 7))

    # Distinct colors for different configurations
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, config in enumerate(configs):
        color = colors[i % len(colors)]

        # Load Data
        cent_runs = get_runs(data_dir, "central", config, lu=None)
        fed_runs = get_runs(data_dir, "federated", config, lu=1)

        if not cent_runs or not fed_runs:
            print(f"Skipping {config} in Plot 4: Missing data")
            continue

        c_train, c_val = extract_curves(cent_runs)
        f_train, f_val = extract_curves(fed_runs)

        cx, _, _ = average_curves(c_train)
        cvx, cvm, cvs = average_curves(c_val)

        fx, _, _ = average_curves(f_train)
        fvx, fvm, fvs = average_curves(f_val)

        # Calculate Scaling Factor based on training step counts
        max_cent_step = (
            cx[-1] if cx is not None else (cvx[-1] if cvx is not None else 0)
        )
        max_fed_step = fx[-1] if fx is not None else (fvx[-1] if fvx is not None else 0)

        scale_factor = 1.0
        if max_cent_step > 0:
            scale_factor = max_fed_step / max_cent_step

        # Apply scaling to Central Validation X
        if cvx is not None:
            cvx = cvx * scale_factor

        # Plot Central (Solid)
        if cvx is not None:
            plt.plot(
                cvx,
                cvm,
                color=color,
                linestyle="-",
                lw=2.5,
                label=f"{config} (Central)",
            )
            plt.fill_between(cvx, cvm - cvs, cvm + cvs, color=color, alpha=0.1)

        # Plot FedAvg (Dashed)
        if fvx is not None:
            plt.plot(
                fvx,
                fvm,
                color=color,
                linestyle="--",
                lw=2.5,
                label=f"{config} (FedAvg)",
            )
            plt.fill_between(fvx, fvm - fvs, fvm + fvs, color=color, alpha=0.1)

    plt.yscale("log")
    plt.title(f"Impact of Model Depth: Central vs FedAvg")
    plt.xlabel("Global Steps (Scaled)")
    plt.ylabel("Validation Perplexity")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"4_multi_config_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Generated Plot 4 (Multi-Config Comparison)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="../data/plots/perplexities")
    parser.add_argument("--config", type=str, default="pure_ssm_1_layer", 
                        help="Which model config to plot (e.g. pure_ssm_1_layer)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Run the 3 requested plots
    plot_central_vs_fed(args.data_dir, args.config, args.out_dir)
    plot_fed_lu_comparison(args.data_dir, args.config, args.out_dir)
    plot_method_comparison(args.data_dir, args.config, args.out_dir)

    configs_to_compare = ["pure_ssm_1_layer", "pure_ssm_2_layer", "pure_ssm_4_layer"]
    plot_multi_config_comparison(args.data_dir, configs_to_compare, args.out_dir)
    
    print("\nPerplexity plots complete.")
