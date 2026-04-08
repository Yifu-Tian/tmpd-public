import os
import pickle
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(INFERENCE_DIR, "..", ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
SHARED_DATA_DIR = os.path.join(RESULTS_ROOT, "benchmark_shared_data")
OUTPUT_PLOTS_DIR = os.path.join(RESULTS_ROOT, "comparison_plots")

# Algorithm display order (fixed for consistent figure layout).
ALGORITHMS = [
    {"suffix": "astar", "name": "Topo-A*"},
    {"suffix": "rrt",   "name": "Topo-RRT"},
    {"suffix": "mpd",   "name": "MPD"},
    {"suffix": "tmpd",  "name": "TMPD (Ours)"}
]

def load_trial_data(trial_idx, suffix):
    filepath = os.path.join(SHARED_DATA_DIR, f"trial_{trial_idx:03d}_{suffix}.pkl")
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None

def plot_1x4_paper_figure(trial_idx, num_segments=5):
    # Load one trial for all methods.
    trial_results = {}
    for algo in ALGORITHMS:
        data = load_trial_data(trial_idx, algo["suffix"])
        trial_results[algo["name"]] = data
    
    valid_data = [d for d in trial_results.values() if d is not None]
    if not valid_data:
        return False
        
    # Use shared env/waypoints from any valid method.
    env_ref = valid_data[0]
    obs_centers = env_ref["obs_centers"]
    obs_types = env_ref["obs_types"]
    obs_dims = env_ref["obs_dims"]
    waypoints = env_ref["waypoints"]
    
    # Build 1x4 comparison canvas.
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    cmap = plt.cm.plasma
    segment_colors = cmap(np.linspace(0.1, 0.9, num_segments))
    
    for idx, algo in enumerate(ALGORITHMS):
        method_name = algo["name"]
        ax = axes[idx]
        data = trial_results[method_name]
        
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle=':', alpha=0.3, zorder=0)
        
        # Emphasize "Ours" in the title.
        weight = 'bold' if 'Ours' in method_name else 'normal'
        ax.set_title(method_name, fontsize=18, weight=weight, pad=10)
        
        if idx == 0: ax.set_ylabel('y (m)', fontsize=14)
        else: ax.set_yticklabels([])
        ax.set_xlabel('x (m)', fontsize=14)

        # Draw obstacles.
        for k, center in enumerate(obs_centers):
            ctype, cdim = obs_types[k], obs_dims[k]
            if ctype == 'sphere':
                ax.add_patch(patches.Circle((center[0], center[1]), cdim[0], facecolor='#d62728', edgecolor='black', lw=1.5, alpha=0.85, zorder=10))
            elif ctype == 'box':
                ax.add_patch(patches.Rectangle((center[0]-cdim[0], center[1]-cdim[1]), 2*cdim[0], 2*cdim[1], facecolor='#7f7f7f', edgecolor='black', lw=1.5, alpha=0.85, zorder=10))

        # Draw waypoints.
        for i, wp in enumerate(waypoints):
            if i == 0:
                ax.plot(wp[0], wp[1], 's', color='#2ca02c', markersize=12, markeredgecolor='black', zorder=30)
                ax.text(wp[0], wp[1]-0.08, 'Start', ha='center', va='top', fontsize=12, weight='bold', color='#2ca02c', zorder=35)
            else:
                ax.plot(wp[0], wp[1], 'o', color='#ff7f0e', markersize=12, markeredgecolor='black', zorder=30)
                ax.text(wp[0], wp[1], str(i), ha='center', va='center', fontsize=10, weight='bold', color='black', zorder=35)

        # Mark panel as missing if this method has no data.
        if data is None:
            ax.text(0, 0, "No Data", ha='center', va='center', fontsize=20, color='red', weight='bold', alpha=0.5)
            continue
            
        history = data["history"]
        fatal_error = data["fatal_error"]
        tangled = data["tangled"]

        # Draw trajectories for completed segments.
        for seg_idx, traj_np in enumerate(history):
            if len(traj_np) < 2: continue
            color = segment_colors[seg_idx]
            
            ax.plot(traj_np[:, 0], traj_np[:, 1], color=color, linewidth=3.5, alpha=0.9, zorder=20, solid_capstyle='round', label=f'Seg {seg_idx+1}' if idx==0 else "")
            
            # Draw direction arrow near 60% of the segment.
            mid = int(len(traj_np) * 0.6)
            if len(traj_np) > 2:
                dx, dy = traj_np[mid+1, 0] - traj_np[mid, 0], traj_np[mid+1, 1] - traj_np[mid, 1]
                norm = math.hypot(dx, dy)
                if norm > 0:
                    ax.arrow(traj_np[mid, 0], traj_np[mid, 1], dx/norm*0.01, dy/norm*0.01, shape='full', lw=0, length_includes_head=True, head_width=0.06, color=color, zorder=25)

        # Highlight failure/tangle states on the panel border.
        if fatal_error:
            failed_seg_idx = len(history) 
            if failed_seg_idx < len(waypoints) - 1:
                failed_wp = waypoints[failed_seg_idx + 1]
                ax.plot(failed_wp[0], failed_wp[1], 'rx', markersize=24, markeredgewidth=4, zorder=40)
                
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
                
        elif tangled:
            for spine in ax.spines.values():
                spine.set_edgecolor('darkorange')
                spine.set_linewidth(3)

    handles, labels = axes[0].get_legend_handles_labels()

    import matplotlib.lines as mlines
    if any(d and d["fatal_error"] for d in trial_results.values()):
        handles.append(mlines.Line2D([], [], color='red', marker='x', markersize=10, markeredgewidth=3, linestyle='None'))
        labels.append('Deadlock')
        
    fig.legend(handles, labels, loc='lower center', ncol=num_segments + 2, fontsize=14, bbox_to_anchor=(0.5, 0.02), frameon=False)
    plt.subplots_adjust(bottom=0.15, left=0.02, right=0.98, top=0.92, wspace=0.08)
    
    out_pdf = os.path.join(OUTPUT_PLOTS_DIR, f"trial_{trial_idx:03d}_1x4_comparison.pdf")
    out_png = os.path.join(OUTPUT_PLOTS_DIR, f"trial_{trial_idx:03d}_1x4_comparison.png")
    plt.savefig(out_pdf, format='pdf', dpi=300)
    plt.savefig(out_png, format='png', dpi=300)
    plt.close(fig)
    return True

if __name__ == '__main__':
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    print(f"Starting 1x4 Figures Generation...")
    
    success_count = 0
    for trial in range(1, 101):
        if plot_1x4_paper_figure(trial):
            print(f" Generated figure for Trial {trial:03d}")
            success_count += 1
            
    print(f"Done! Generated {success_count} comparison plots in '{OUTPUT_PLOTS_DIR}'.")
