"""terrain_visualizer.py

Quantum terrain visualizer for the Kingston 46-window hardware run.

Reads output/prime/20260815_204703/results_hw.json and renders the
probability distributions across all 46 windows as evolving topographic
landscape:

  X — basis states |0000⟩ through |1111⟩ (16 columns)
  Y — window index 0–45 (position in prime sequence)
  Z — probability mass (elevation / color)

Outputs:
  terrain_visualizer/terrain_static.png   — full 46-window topographic map
  terrain_visualizer/terrain_animated.gif — frame-by-frame reveal, 46 frames

Data source: output/prime/20260815_204703/results_hw.json
No hardware calls — reads recorded counts only.

Run: python terrain_visualizer.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LightSource, LinearSegmentedColormap

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).parent
DATA_FILE  = REPO_ROOT / "output/prime/20260815_204703/results_hw.json"
SHOTS      = 8_192
N_STATES   = 16
STATE_LABELS = [f"|{i:04b}⟩" for i in range(N_STATES)]

# ── Colormap — dark navy valleys → deep red peaks → gold summits ───────────────

TERRAIN_CMAP = LinearSegmentedColormap.from_list(
    "quantum_terrain",
    [
        "#000814",  # near-black (0 prob)
        "#03045e",  # deep navy
        "#5a189a",  # violet
        "#9b2226",  # dark red
        "#d62828",  # red
        "#f48c06",  # amber
        "#ffd60a",  # gold (max prob)
    ],
)

# Dark background colors
BG   = "#0a0e18"
FG   = "#e8eaf6"
MUT  = "#7986cb"
GRN  = "#66bb6a"
RED  = "#ef5350"
GOLD = "#ffd60a"
PRP  = "#ce93d8"

# ── Data loading ───────────────────────────────────────────────────────────────

def load_terrain() -> tuple[np.ndarray, list[dict], dict]:
    data = json.loads(DATA_FILE.read_text())
    per_window = data["per_window"]
    final_counts = data["final_counts"]
    terrain = np.zeros((46, N_STATES))

    for w in per_window:
        w_idx = w["w"]
        if w_idx == 45:
            # Full 16-state distribution available
            probs = np.zeros(N_STATES)
            for bs, cnt in final_counts.items():
                probs[int(bs, 2)] = cnt / SHOTS
        else:
            # Reconstruct: top_counts are the 8 highest; distribute
            # remaining shots equally among the 8 unlisted states
            probs = np.zeros(N_STATES)
            assigned = 0
            for bs, cnt in w["top_counts"].items():
                probs[int(bs, 2)] = cnt
                assigned += cnt
            remaining = SHOTS - assigned
            missing = [i for i in range(N_STATES) if probs[i] == 0]
            if missing and remaining > 0:
                share = remaining / len(missing)
                for i in missing:
                    probs[i] = share
            probs /= SHOTS
        terrain[w_idx] = probs

    return terrain, per_window, data

# ── Static topographic render ──────────────────────────────────────────────────

def plot_static(terrain: np.ndarray, per_window: list[dict], ts: str, out_dir: Path) -> Path:
    mi_vals = [w["mi"] for w in per_window]
    gaps_list = [w["gaps"] for w in per_window]
    peak_win   = int(np.argmax(mi_vals))
    trough_win = int(np.argmin(mi_vals))

    # Smooth for hillshading (interpolate to higher res grid)
    from scipy.ndimage import zoom
    terrain_hires = zoom(terrain, (4, 4), order=3)
    terrain_hires = np.clip(terrain_hires, 0, None)

    ls = LightSource(azdeg=270, altdeg=45)
    rgb = ls.shade(
        terrain_hires,
        cmap=TERRAIN_CMAP,
        vert_exag=6.0,
        blend_mode="soft",
        vmin=0,
        vmax=terrain.max(),
    )

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Hillshaded terrain
    ax.imshow(rgb, aspect="auto", origin="upper",
              extent=[-0.5, N_STATES - 0.5, 45.5, -0.5])

    # Contour lines on original-resolution terrain
    X = np.arange(N_STATES)
    Y = np.arange(46)
    levels = np.linspace(terrain.min() + 0.005, terrain.max(), 10)
    ax.contour(X, Y, terrain, levels=levels,
               colors="white", alpha=0.25, linewidths=0.6)

    # Annotate peak MI window
    ax.annotate(
        f"Peak MI\nw{peak_win}: {mi_vals[peak_win]:.3f}\n{gaps_list[peak_win]}",
        xy=(N_STATES - 0.5, peak_win),
        xytext=(N_STATES + 0.4, peak_win),
        fontsize=7.5, color=GOLD,
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.0),
        ha="left", va="center",
    )

    # Annotate trough MI window
    ax.annotate(
        f"Trough MI\nw{trough_win}: {mi_vals[trough_win]:.3f}\n{gaps_list[trough_win]}",
        xy=(N_STATES - 0.5, trough_win),
        xytext=(N_STATES + 0.4, trough_win),
        fontsize=7.5, color=PRP,
        arrowprops=dict(arrowstyle="->", color=PRP, lw=1.0),
        ha="left", va="center",
    )

    # Annotate prediction window (45)
    ax.annotate(
        "Prediction\nw45: gap→233",
        xy=(N_STATES - 0.5, 45),
        xytext=(N_STATES + 0.4, 45),
        fontsize=7.5, color=GRN,
        arrowprops=dict(arrowstyle="->", color=GRN, lw=1.0),
        ha="left", va="center",
    )

    # MI heat strip on left margin
    mi_arr = np.array(mi_vals).reshape(-1, 1)
    mi_ax = ax.inset_axes([-0.075, 0, 0.05, 1])
    mi_ax.imshow(mi_arr, aspect="auto", origin="upper",
                 cmap="plasma", vmin=0, vmax=max(mi_vals))
    mi_ax.set_xticks([])
    mi_ax.set_yticks([])
    mi_ax.set_title("MI", color=MUT, fontsize=7, pad=3)

    # Axes styling
    ax.set_xlim(-0.5, N_STATES - 0.5)
    ax.set_ylim(45.5, -0.5)
    ax.set_xticks(range(N_STATES))
    ax.set_xticklabels(STATE_LABELS, rotation=45, ha="right", fontsize=7.5, color=MUT)
    ax.set_yticks(range(0, 46, 5))
    ax.set_yticklabels([f"w{i}" for i in range(0, 46, 5)], fontsize=8, color=MUT)
    ax.tick_params(colors=MUT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")

    ax.set_xlabel("Basis state", color=MUT, fontsize=9, labelpad=8)
    ax.set_ylabel("Window (position in prime sequence)", color=MUT, fontsize=9)
    ax.set_title(
        "Quantum Terrain — ibm_kingston Hardware, 46 Recurrent Windows\n"
        "Prime gap sequence encoded as probability landscape  |  "
        "Red peaks = high probability, navy valleys = low",
        color=FG, fontsize=11, pad=12,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=TERRAIN_CMAP,
                                norm=plt.Normalize(0, terrain.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.18)
    cbar.set_label("Probability", color=MUT, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=MUT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUT, fontsize=8)

    fig.tight_layout()
    p = out_dir / "terrain_static.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return p


# ── Animated GIF ───────────────────────────────────────────────────────────────

def make_animation(terrain: np.ndarray, per_window: list[dict], ts: str, out_dir: Path) -> Path:
    mi_vals  = [w["mi"] for w in per_window]
    gaps_list = [w["gaps"] for w in per_window]
    lmax_list = [w["local_max"] for w in per_window]
    modes    = [w["mode"] for w in per_window]

    vmax = terrain.max()

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    def draw_frame(i: int) -> None:
        ax.cla()
        ax.set_facecolor(BG)

        # Revealed terrain: windows 0..i at full brightness
        # Unrevealed: windows i+1..45 very dim
        display = terrain.copy()
        if i < 45:
            display[i + 1 :] *= 0.08   # dim but not invisible — shows shape

        ax.imshow(
            display,
            aspect="auto",
            origin="upper",
            cmap=TERRAIN_CMAP,
            vmin=0,
            vmax=vmax,
            interpolation="bicubic",
        )

        # Contours on revealed portion only
        revealed = terrain.copy()
        revealed[i + 1 :] = 0
        X = np.arange(N_STATES)
        Y = np.arange(46)
        lvls = np.linspace(0.02, vmax, 7)
        ax.contour(X, Y, revealed, levels=lvls,
                   colors="white", alpha=0.2, linewidths=0.5)

        # Scanline at current window
        ax.axhline(i, color=GOLD, linewidth=2.0, alpha=0.85, zorder=5)

        # Highlight current row's bar chart inline
        row_probs = terrain[i]
        bar_y = i + 0.38
        for s in range(N_STATES):
            h = row_probs[s] * 2.5   # scale height to row units
            rect = plt.Rectangle(
                (s - 0.4, bar_y - h), 0.8, h,
                color=TERRAIN_CMAP(row_probs[s] / vmax),
                alpha=0.7, zorder=6,
            )
            ax.add_patch(rect)

        # Text info box
        info = (
            f"Window {i:02d}/45   gaps={gaps_list[i]}   lmax={lmax_list[i]}\n"
            f"MI={mi_vals[i]:.4f} bits   mode=|{modes[i]}⟩"
        )
        ax.text(
            0.01, 0.02, info,
            transform=ax.transAxes,
            fontsize=8.5, color=FG,
            bbox=dict(facecolor="#0a0e18cc", edgecolor=GOLD, linewidth=0.8, pad=4),
            va="bottom",
        )

        # Axes
        ax.set_xlim(-0.5, N_STATES - 0.5)
        ax.set_ylim(45.5, -0.5)
        ax.set_xticks(range(N_STATES))
        ax.set_xticklabels(STATE_LABELS, rotation=45, ha="right", fontsize=7, color=MUT)
        ax.set_yticks(range(0, 46, 5))
        ax.set_yticklabels([f"w{j}" for j in range(0, 46, 5)], fontsize=7.5, color=MUT)
        ax.tick_params(colors=MUT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e293b")
        ax.set_xlabel("Basis state", color=MUT, fontsize=8)
        ax.set_ylabel("Window index", color=MUT, fontsize=8)
        ax.set_title(
            f"Quantum Terrain — ibm_kingston  |  frame {i+1}/46",
            color=FG, fontsize=10,
        )

    anim = FuncAnimation(fig, draw_frame, frames=46, interval=250)
    p = out_dir / "terrain_animated.gif"
    writer = PillowWriter(fps=4)
    anim.save(str(p), writer=writer, dpi=100)
    plt.close(fig)
    return p


# ── Auto-commit ────────────────────────────────────────────────────────────────

def auto_commit_push(ts: str) -> None:
    subprocess.run(
        ["git", "add", f"output/prime/{ts}/"],
        check=True, cwd=REPO_ROOT,
    )
    result = subprocess.run(
        ["git", "commit", "-m",
         f"Terrain visualizer {ts} — static PNG + 46-frame animated GIF from Kingston hardware run"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  Committed output/prime/{ts}/")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed.")
    else:
        print(f"  Commit skipped: {result.stdout.strip()}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "output" / "prime" / ts / "terrain_visualizer"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {DATA_FILE.relative_to(REPO_ROOT)}")
    terrain, per_window, data = load_terrain()
    print(f"  Terrain shape: {terrain.shape}  max prob: {terrain.max():.4f}")
    mi_vals = [w["mi"] for w in per_window]
    print(f"  Peak MI: w{np.argmax(mi_vals)} = {max(mi_vals):.4f}")
    print(f"  Trough MI: w{np.argmin(mi_vals)} = {min(mi_vals):.4f}")

    print("\nRendering static terrain map...")
    static_p = plot_static(terrain, per_window, ts, out_dir)
    print(f"  → {static_p.relative_to(REPO_ROOT)}")

    print("Rendering animated GIF (46 frames @ 4fps)...")
    anim_p = make_animation(terrain, per_window, ts, out_dir)
    print(f"  → {anim_p.relative_to(REPO_ROOT)}")

    auto_commit_push(ts)


if __name__ == "__main__":
    main()
