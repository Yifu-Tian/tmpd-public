import math

import matplotlib.pyplot as plt
import numpy as np


def render_segmented_trial_plot(
    env,
    waypoints_np,
    history_trajs,
    trial_idx,
    method_label,
    status_txt,
    title_color,
    output_path,
    taut_traj=None,
    is_tangled=False,
    failed_goal=None,
    failed_goal_label=None,
    failed_traj=None,
    dpi=150,
):
    """Render and save one benchmark trial figure."""
    fig, ax = plt.subplots(figsize=(10, 10))
    env.render(ax)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(
        f"{method_label} - Trial {trial_idx} [{status_txt}]",
        fontsize=16,
        weight='bold',
        color=title_color,
    )

    cmap = plt.cm.winter
    segment_colors = cmap(np.linspace(0.0, 1.0, max(1, len(waypoints_np) - 1)))

    for k, wp in enumerate(waypoints_np):
        if k == 0:
            ax.plot(
                wp[0], wp[1], 's',
                color='green',
                markersize=12,
                markeredgecolor='black',
                zorder=20,
            )
            ax.text(
                wp[0], wp[1] - 0.08, 'S',
                color='green',
                ha='center',
                va='top',
                weight='bold',
                zorder=21,
            )
        else:
            ax.plot(
                wp[0], wp[1], 'o',
                color='gold',
                markersize=12,
                markeredgecolor='black',
                zorder=20,
            )
            ax.text(
                wp[0], wp[1], str(k),
                color='black',
                ha='center',
                va='center',
                weight='bold',
                zorder=21,
            )

    for seg_idx, traj in enumerate(history_trajs):
        if len(traj) < 2:
            continue

        c = segment_colors[min(seg_idx, len(segment_colors) - 1)]
        ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=3.5, alpha=0.85, label=f'Seg {seg_idx+1}')

        mid = int(len(traj) * 0.6)
        if len(traj) > 2:
            dx, dy = traj[mid + 1, 0] - traj[mid, 0], traj[mid + 1, 1] - traj[mid, 1]
            norm = math.hypot(dx, dy)
            if norm > 0:
                ax.arrow(
                    traj[mid, 0],
                    traj[mid, 1],
                    dx / norm * 0.001,
                    dy / norm * 0.001,
                    shape='full',
                    lw=0,
                    length_includes_head=True,
                    head_width=0.02,
                    head_length=0.03,
                    color=c,
                    zorder=25,
                )

    if failed_traj is not None:
        ax.plot(
            failed_traj[:, 0],
            failed_traj[:, 1],
            color='red',
            linewidth=2.5,
            linestyle='--',
            alpha=0.9,
            zorder=18,
            label='Collision Trajectory',
        )
        ax.plot(
            failed_traj[-1, 0],
            failed_traj[-1, 1],
            'rx',
            markersize=12,
            markeredgewidth=3,
            zorder=19,
        )

    if failed_goal is not None:
        kwargs = {}
        if failed_goal_label is not None:
            kwargs["label"] = failed_goal_label
        ax.plot(
            failed_goal[0],
            failed_goal[1],
            'rx',
            markersize=20,
            markeredgewidth=4,
            zorder=25,
            **kwargs,
        )

    if taut_traj is not None:
        if is_tangled:
            ax.plot(
                taut_traj[:, 0],
                taut_traj[:, 1],
                color='red',
                linewidth=3.5,
                linestyle='--',
                alpha=0.9,
                zorder=15,
                label='Tangled Taut Curve',
            )
        else:
            ax.plot(
                taut_traj[:, 0],
                taut_traj[:, 1],
                color='darkorange',
                linewidth=2.5,
                linestyle=':',
                alpha=0.6,
                zorder=15,
                label='Safe Taut Curve',
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
