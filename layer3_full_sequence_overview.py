"""layer3_full_sequence_overview.py

Single full-context visual (hypotheses/regime_internal_wave_structure.md):
the entire ~20,000-point raw gap sequence in one wide plot, with rolling
std (K=100, same window used throughout this file's Layer 3 work) overlaid
on a twin y-axis, and both known changepoint sets marked for comparison --
the 39 gap-space changepoints from layer3_20k_scaleup.py
(output/prime/20260818_015045/results.json) and the original 3 MI-space
changepoints (1529, 2501, 4211) from regime_fit_5k.py. See CLAUDE.md's
"Two changepoint sets" note for why these are different things measured
different ways, not two versions of the same detection.

No new statistics or claims -- purely a visual overview.

Run: python layer3_full_sequence_overview.py
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).parent
GAPS_CACHE_PATH = REPO_ROOT / "data/primes_20000.json"
GAP_SPACE_CHANGEPOINTS_SOURCE = REPO_ROOT / "output/prime/20260818_015045/results.json"
MI_SPACE_CHANGEPOINTS = [1529, 2501, 4211]

ROLLING_STD_WINDOW = 100

OUT_ROOT = REPO_ROOT / "output" / "prime"


def load_full_gaps() -> np.ndarray:
    with open(GAPS_CACHE_PATH) as f:
        cache = json.load(f)
    gaps = np.array(cache["gaps"])
    assert len(gaps) == cache["n_gaps"]
    return gaps


def load_gap_space_changepoints() -> list[int]:
    with open(GAP_SPACE_CHANGEPOINTS_SOURCE) as f:
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
    msg = f"analysis: layer3 full sequence overview {ts} -- 20k gaps, 39 gap-space + 3 MI-space changepoints"
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
    gap_space_cps = load_gap_space_changepoints()
    print(f"Loaded {len(full_gaps)} raw gaps from {GAPS_CACHE_PATH.relative_to(REPO_ROOT)}")
    print(f"Loaded {len(gap_space_cps)} gap-space changepoints from "
          f"{GAP_SPACE_CHANGEPOINTS_SOURCE.relative_to(REPO_ROOT)}")
    print(f"MI-space changepoints (hardcoded, from regime_fit_5k.py): {MI_SPACE_CHANGEPOINTS}")

    vol = rolling_std(full_gaps, ROLLING_STD_WINDOW)
    vol_x = np.arange(len(vol)) + ROLLING_STD_WINDOW / 2

    fig, ax_seq = plt.subplots(figsize=(24, 6))
    ax_seq.plot(np.arange(len(full_gaps)), full_gaps, color="#4c72b0", lw=0.4, alpha=0.7)
    ax_vol = ax_seq.twinx()
    ax_vol.plot(vol_x, vol, color="black", lw=1.0)

    for cp in gap_space_cps:
        ax_seq.axvline(cp, color="#94a3b8", lw=0.6, alpha=0.35, zorder=1)
    for cp in MI_SPACE_CHANGEPOINTS:
        ax_seq.axvline(cp, color="#d1495b", lw=1.6, ls="--", alpha=0.9, zorder=2)

    ax_seq.set_xlabel("gap index (prime index)")
    ax_seq.set_ylabel("gap size", color="#4c72b0")
    ax_vol.set_ylabel(f"rolling std (K={ROLLING_STD_WINDOW})", color="black")
    ax_seq.set_xlim(0, len(full_gaps))
    ax_seq.set_title(
        f"Full raw gap sequence -- {len(full_gaps)} gaps, rolling-std overlay, "
        f"both changepoint sets marked"
    )

    legend_handles = [
        Line2D([0], [0], color="#4c72b0", lw=1.5, alpha=0.8, label="raw gap sequence"),
        Line2D([0], [0], color="black", lw=1.5, label=f"rolling std (K={ROLLING_STD_WINDOW})"),
        Line2D([0], [0], color="#94a3b8", lw=1.2, alpha=0.6, label=f"gap-space changepoints (n={len(gap_space_cps)})"),
        Line2D([0], [0], color="#d1495b", lw=1.8, ls="--", label="MI-space changepoints (n=3: 1529, 2501, 4211)"),
    ]
    ax_seq.legend(handles=legend_handles, fontsize=9, loc="upper left")

    fig.tight_layout()

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer3_full_sequence_overview.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {out_path.relative_to(REPO_ROOT)}")

    auto_commit_push(out_dir, ts)


if __name__ == "__main__":
    main()
