"""layer3_regime_wave_gallery.py

Visual companion to the 40-Regime Characterization
(hypotheses/regime_internal_wave_structure.md, "## 40-Regime
Characterization"): that check found mean/variance climb significantly with
position (expected, PNT) while skew/kurtosis don't (n=40, permutation
tested). This script adds no new statistical test -- it's a qualitative
look at 10 of the 40 regimes' actual shapes, spread across the sequence,
so the numbers in that table can be checked against what the waveforms
themselves look like.

Reuses the same 39 changepoints (output/prime/20260818_015045/results.json)
and the same 40-regime bounds/stats (output/prime/20260818_020038/results.json)
already computed -- nothing here is re-detected or recomputed, only
re-plotted.

Two outputs:
  1. A 2x5 grid, one subplot per selected regime, raw gap sequence, titled
     with regime index, position range, and its already-computed mean/
     variance/skew/kurtosis.
  2. One overlay plot of all 10 regimes, resampled to a common length and
     z-scored -- the same normalize-for-length-and-amplitude style used in
     layer3_regime_overlay.py's cross-regime self-similarity test -- so
     it's directly comparable to that earlier visual.

Run: python layer3_regime_wave_gallery.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"
CHANGEPOINTS_SOURCE = REPO_ROOT / "output/prime/20260818_015045/results.json"
REGIME_STATS_SOURCE = REPO_ROOT / "output/prime/20260818_020038/results.json"

N_SELECTED = 10
N_RESAMPLE = 500  # matches layer3_regime_overlay.py's resample_and_zscore

OUT_ROOT = REPO_ROOT / "output" / "prime"


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    gaps = np.array(cache["gaps"])
    assert len(gaps) == cache["n_gaps"]
    return gaps


def load_changepoint_positions() -> list[int]:
    with open(CHANGEPOINTS_SOURCE) as f:
        data = json.load(f)
    return [c["position"] for c in data["changepoints"]]


def load_regime_stats() -> list[dict]:
    with open(REGIME_STATS_SOURCE) as f:
        data = json.load(f)
    return data["regimes"]


def resample_and_zscore(y: np.ndarray, n: int) -> np.ndarray:
    """Same normalization as layer3_regime_overlay.py: linear-interpolate to
    a common length, then z-score -- compares shape, not magnitude or
    duration."""
    x_orig = np.linspace(0.0, 1.0, len(y))
    x_new = np.linspace(0.0, 1.0, n)
    y_rs = np.interp(x_new, x_orig, y)
    std = y_rs.std()
    return (y_rs - y_rs.mean()) / std if std > 0 else y_rs - y_rs.mean()


def select_spread_indices(n_regimes: int, n_selected: int) -> list[int]:
    raw = np.linspace(0, n_regimes - 1, n_selected)
    idx = sorted(set(int(round(v)) for v in raw))
    # linspace rounding can collide on small ranges; not expected at
    # n_regimes=40, n_selected=10, but guard and report if it ever does.
    if len(idx) < n_selected:
        remaining = [i for i in range(n_regimes) if i not in idx]
        idx = sorted(idx + remaining[: n_selected - len(idx)])
    return idx


def auto_commit_push(out_dir: Path, selected: list[int], ts: str) -> None:
    msg = f"analysis: layer3 regime wave gallery {ts} -- 10 regimes selected (indices {selected})"
    subprocess.run(["git", "add", str(out_dir.relative_to(REPO_ROOT))], check=True, cwd=REPO_ROOT)
    commit = subprocess.run(["git", "commit", "-m", msg], cwd=REPO_ROOT, capture_output=True, text=True)
    if commit.returncode == 0:
        print(f"\n  Committed: {out_dir.relative_to(REPO_ROOT)}")
        subprocess.run(["git", "push"], check=True, cwd=REPO_ROOT)
        print("  Pushed to remote.")
    else:
        print(f"\n  Git commit skipped: {commit.stdout.strip()}")


def main() -> None:
    full_gaps = load_full_gaps()
    positions_cp = load_changepoint_positions()
    regime_stats = load_regime_stats()
    print(f"Loaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")
    print(f"Loaded {len(positions_cp)} changepoints from {CHANGEPOINTS_SOURCE.relative_to(REPO_ROOT)}")
    print(f"Loaded {len(regime_stats)} regime stat records from {REGIME_STATS_SOURCE.relative_to(REPO_ROOT)}")

    bounds = [(0, positions_cp[0])]
    bounds += [(positions_cp[i], positions_cp[i + 1]) for i in range(len(positions_cp) - 1)]
    bounds += [(positions_cp[-1], len(full_gaps))]
    assert len(bounds) == 40
    assert [tuple(regime_stats[i]["bounds"]) for i in range(40)] == bounds, (
        "Regime bounds recomputed here don't match the saved 40-regime characterization -- "
        "the two runs would be describing different regimes."
    )

    selected = select_spread_indices(40, N_SELECTED)
    print(f"\nSelected {len(selected)} regimes, spread across the full range: {selected}")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Gallery: one subplot per selected regime ────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    colors = plt.cm.tab10.colors
    for ax, ridx, color in zip(axes.flat, selected, colors, strict=True):
        a, b = bounds[ridx]
        x = full_gaps[a:b]
        stats = regime_stats[ridx]
        ax.plot(np.arange(len(x)), x, color=color, lw=0.8)
        ax.set_title(
            f"regime {ridx}  [{a}, {b})\n"
            f"mean={stats['mean']:.2f}  var={stats['variance']:.2f}\n"
            f"skew={stats['skew']:.3f}  kurt={stats['kurtosis_excess_internal']:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("index within regime", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Regime wave gallery -- 10 of 40 regimes, spread across the sequence [{ts}]", fontsize=13)
    fig.tight_layout()
    gallery_path = out_dir / "layer3_regime_wave_gallery.png"
    fig.savefig(gallery_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {gallery_path.relative_to(REPO_ROOT)}")

    # ── Overlay: same normalize-for-length-and-amplitude style as
    # layer3_regime_overlay.py's cross-regime self-similarity test ──────
    fig, ax = plt.subplots(figsize=(13, 7))
    x_norm = np.linspace(0.0, 1.0, N_RESAMPLE)
    for ridx, color in zip(selected, colors, strict=True):
        a, b = bounds[ridx]
        x = full_gaps[a:b]
        y_norm = resample_and_zscore(x, N_RESAMPLE)
        ax.plot(x_norm, y_norm, color=color, lw=1.1, alpha=0.85, label=f"regime {ridx} [{a},{b})")
    ax.set_xlabel("fractional position within regime")
    ax.set_ylabel("z-scored gap")
    ax.set_title(f"Overlay -- 10 regimes, length+amplitude normalized (n_resample={N_RESAMPLE}) [{ts}]")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    overlay_path = out_dir / "layer3_regime_wave_overlay.png"
    fig.savefig(overlay_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {overlay_path.relative_to(REPO_ROOT)}")

    auto_commit_push(out_dir, selected, ts)


if __name__ == "__main__":
    main()
