# src/utils/visualize.py
#
# Visualization utilities for the BEV pipeline outputs.
# Provides richer BEV plots than the inline pipeline.py version.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_bev_panel(image, depth, grid, detections, object_positions,
                   save_path=None, title_suffix=""):
    """
    3-panel visualization: camera view | depth map | BEV occupancy grid.

    Args:
        image:            BGR image
        depth:            float32 depth map (H x W), metres
        grid:             OccupancyGrid instance
        detections:       list of [x1,y1,x2,y2,label,conf]
        object_positions: list of (col, row, label, conf) in grid coords
        save_path:        if given, save figure here
        title_suffix:     appended to each subplot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d")

    # ── Panel 1: Camera + Detections ──
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    for det in detections:
        x1, y1, x2, y2, label, conf = det
        col = "lime" if label in ["car", "truck", "bus"] else "cyan"
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor=col, linewidth=2)
        axes[0].add_patch(rect)
        axes[0].text(x1, max(y1-6, 0), f"{label} {conf:.2f}",
                     color=col, fontsize=8.5, fontweight="bold",
                     bbox=dict(facecolor="black", alpha=0.4, pad=1, linewidth=0))
    axes[0].set_title(f"Camera View + Detections {title_suffix}",
                      color="white", fontsize=11)
    axes[0].axis("off")

    # ── Panel 2: Depth Map ──
    # FIX: show the actual depth range properly — no more binary blue/yellow
    vmin, vmax = float(np.percentile(depth, 2)), float(np.percentile(depth, 98))
    im2 = axes[1].imshow(depth, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Depth Map (metres) {title_suffix}", color="white", fontsize=11)
    axes[1].axis("off")
    cbar = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("depth (m)", color="white")

    # ── Panel 3: BEV Occupancy Grid ──
    g = grid.grid.copy()
    axes[2].imshow(g, cmap="inferno", origin="lower", vmin=0, vmax=1)

    # Draw ego-vehicle marker at the bottom centre
    ego_col = grid.N // 2
    ego_row = 0
    axes[2].plot(ego_col, ego_row, "w^", markersize=10, label="ego")

    # Label detected objects on the grid
    for col, row, label, conf in object_positions:
        axes[2].plot(col, row, "wo", markersize=6, markerfacecolor="none",
                     markeredgewidth=1.5)
        axes[2].text(col + 3, row + 3, f"{label}\n{conf:.2f}",
                     color="white", fontsize=7.5, fontweight="bold")

    # Grid axes: show real-world distances
    res = grid.res
    size = grid.size
    tick_step_cells = int(5.0 / res)  # every 5 metres
    xticks = np.arange(0, grid.N + 1, tick_step_cells)
    xlabels = [f"{(x * res - size/2):.0f}m" for x in xticks]
    yticks = np.arange(0, grid.N + 1, tick_step_cells)
    ylabels = [f"{y * res:.0f}m" for y in yticks]

    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(xlabels, color="white", fontsize=7)
    axes[2].set_yticks(yticks)
    axes[2].set_yticklabels(ylabels, color="white", fontsize=7)
    axes[2].set_xlabel("X lateral", color="white")
    axes[2].set_ylabel("Z forward (m)", color="white")
    axes[2].set_title(f"BEV Occupancy Grid {title_suffix}", color="white", fontsize=11)
    axes[2].tick_params(colors="white")
    for spine in axes[2].spines.values():
        spine.set_edgecolor("white")

    plt.tight_layout(pad=1.5)
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {save_path}")
    plt.show()
    plt.close(fig)


def plot_depth_comparison(depth_raw, depth_fixed, save_path=None):
    """
    Side-by-side comparison of old (broken) vs new (fixed) depth maps.
    Useful for validating the depth estimator fix.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(depth_raw, cmap="plasma")
    ax1.set_title("Old depth (broken — binary sky/foreground)")
    ax1.axis("off")

    ax2.imshow(depth_fixed, cmap="plasma",
               vmin=np.percentile(depth_fixed, 2),
               vmax=np.percentile(depth_fixed, 98))
    ax2.set_title("Fixed depth (smooth gradient, metric scale)")
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def save_grid_heatmap(grid, save_path, cmap="inferno"):
    """
    Save just the BEV occupancy grid as a standalone heatmap image.
    Each pixel = one grid cell, value = occupancy probability [0, 1].
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid.grid, cmap=cmap, origin="lower", vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid heatmap saved → {save_path}")