"""layer3_regime_wave_gallery_detailed.py

Full-size companion to layer3_regime_wave_gallery.py's thumbnail gallery
(hypotheses/regime_internal_wave_structure.md, "## Visual Gallery: 10
Regimes"): same 10 regimes ([0, 4, 9, 13, 17, 22, 26, 30, 35, 39]), same
data sources, but each panel now matches the exact plot style used for the
original 3-regime characterization panels
(layer3_regime_characterization.py's plot_regime_panels, left column) --
raw gap sequence on the primary y-axis, rolling std (K=100) overlaid on a
twin y-axis, index-within-regime on the x-axis -- rather than the smaller
thumbnail style used in the first gallery script.

Additive: does not replace or modify the existing thumbnail gallery grid
or the normalized overlay plot from layer3_regime_wave_gallery.py.

Run: python layer3_regime_wave_gallery_detailed.py
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

SELECTED = [0, 4, 9, 13, 17, 22, 26, 30, 35, 39]  # same 10 regimes as layer3_regime_wave_gallery.py
ROLLING_STD_WINDOW = 100  # same K as layer3_regime_characterization.py

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


def rolling_std(x: np.ndarray, k: int) -> np.ndarray:
    """Leading-window rolling std: value at position i is std(x[i:i+k]) --
    identical to layer3_regime_characterization.py's rolling_std."""
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x**2, 0, 0.0))
    mean = (c1[k:] - c1[:-k]) / k
    mean_sq = (c2[k:] - c2[:-k]) / k
    return np.sqrt(np.clip(mean_sq - mean**2, 0.0, None))


def auto_commit_push(out_dir: Path, ts: str) -> None:
    msg = f"analysis: layer3 regime wave gallery detailed panels {ts} -- 10 full-size raw+volatility panels"
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
    print(f"Loaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")
    print(f"Loaded {len(positions_cp)} changepoints from {CHANGEPOINTS_SOURCE.relative_to(REPO_ROOT)}")

    bounds = [(0, positions_cp[0])]
    bounds += [(positions_cp[i], positions_cp[i + 1]) for i in range(len(positions_cp) - 1)]
    bounds += [(positions_cp[-1], len(full_gaps))]
    assert len(bounds) == 40

    print(f"Using the same 10 selected regimes as layer3_regime_wave_gallery.py: {SELECTED}")

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(5, 2, figsize=(16, 22))
    for ax_seq, ridx, color in zip(axes.flat, SELECTED, colors, strict=True):
        a, b = bounds[ridx]
        x = full_gaps[a:b]
        vol = rolling_std(x, ROLLING_STD_WINDOW)
        vol_x = np.arange(len(vol)) + ROLLING_STD_WINDOW / 2

        ax_seq.plot(np.arange(len(x)), x, color=color, lw=0.6, alpha=0.6, label="raw gap")
        ax_vol = ax_seq.twinx()
        ax_vol.plot(vol_x, vol, color="black", lw=1.4, label=f"rolling std (K={ROLLING_STD_WINDOW})")

        ax_seq.set_title(f"regime {ridx}  [{a}, {b})  (n={len(x)}) -- raw sequence + volatility overlay")
        ax_seq.set_xlabel("index within regime")
        ax_seq.set_ylabel("gap size", color=color)
        ax_vol.set_ylabel("rolling std", color="black")

        lines1, labels1 = ax_seq.get_legend_handles_labels()
        lines2, labels2 = ax_vol.get_legend_handles_labels()
        ax_seq.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    fig.suptitle(f"Regime wave gallery -- detailed panels, 10 of 40 regimes [{ts}]", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "layer3_regime_wave_gallery_detailed.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path.relative_to(REPO_ROOT)}")

    auto_commit_push(out_dir, ts)


if __name__ == "__main__":
    main()
