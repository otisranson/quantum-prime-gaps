"""build_prime_cache.py

Part A data refactor: this repo had no standalone primes/gap cache file.
The 5000-prime dataset was generated inline by terrain_5000primes.py's own
sieve() each run (quantum circuit included), and eight classical analysis
scripts each independently reconstructed the raw gap sequence from
output/prime/20260816_010716/terrain_5000primes/results_5000primes.json's
per_window data via the same repeated boilerplate:
`[r["gaps"][0] for r in per_window] + per_window[-1]["gaps"][1:]`.

This script extracts that into two standalone cache files:

  data/primes_5000.json  -- first 5000 primes + their 4999 gaps
  data/primes_20000.json -- first 20000 primes + their 19999 gaps

Correctness: the 5000-prime cache is independently verified against the
existing quantum-run JSON's reconstructed gap sequence -- two different
generation paths (an independent sieve here vs. terrain_5000primes.py's own
sieve baked into that run's output) must produce byte-identical numbers
before this cache is trusted as the new source of truth for every script
that reads it.

Run: python build_prime_cache.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
VERIFY_SOURCE = REPO_ROOT / "output/prime/20260816_010716/terrain_5000primes/results_5000primes.json"


def sieve_primes(count: int) -> list[int]:
    """Sieve of Eratosthenes, doubling the search limit until enough primes
    are found. Same pattern already used in several scripts in this repo
    (quantum_prime_gaps/quantum_prime_gaps.py, mi_landscape_25groups.py,
    terrain_5000primes.py) -- not imported from any of them, kept
    self-contained per this repo's standalone-script convention."""
    limit = 10
    while True:
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        primes = [i for i, p in enumerate(is_p) if p]
        if len(primes) >= count:
            return primes[:count]
        limit *= 2


def build_cache(n_primes: int, out_path: Path) -> dict:
    primes = sieve_primes(n_primes)
    gaps = [primes[i + 1] - primes[i] for i in range(len(primes) - 1)]
    payload = {
        "n_primes": len(primes),
        "n_gaps": len(gaps),
        "first_prime": primes[0],
        "last_prime": primes[-1],
        "primes": primes,
        "gaps": gaps,
    }
    out_path.write_text(json.dumps(payload))
    return payload


def verify_5000_cache_against_quantum_run(cache: dict) -> None:
    with open(VERIFY_SOURCE) as f:
        data = json.load(f)
    per_window = data["per_window"]
    reconstructed = [r["gaps"][0] for r in per_window] + per_window[-1]["gaps"][1:]
    assert cache["gaps"] == reconstructed, (
        "Independently sieved 5000-prime gap cache does not match the gap sequence "
        "reconstructed from the existing quantum-run JSON -- something is inconsistent "
        "and every script reading the cache would silently get different numbers."
    )
    print(f"Verified: independently-sieved cache matches quantum-run reconstruction exactly "
          f"({len(cache['gaps'])} gaps).")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    cache_5000 = build_cache(5000, DATA_DIR / "primes_5000.json")
    verify_5000_cache_against_quantum_run(cache_5000)
    print(f"data/primes_5000.json: {cache_5000['n_primes']} primes, {cache_5000['n_gaps']} gaps, "
          f"range [{cache_5000['first_prime']}, {cache_5000['last_prime']}]")

    cache_20000 = build_cache(20000, DATA_DIR / "primes_20000.json")
    print(f"data/primes_20000.json: {cache_20000['n_primes']} primes, {cache_20000['n_gaps']} gaps, "
          f"range [{cache_20000['first_prime']}, {cache_20000['last_prime']}]")

    print()
    for name in ["primes_5000.json", "primes_20000.json"]:
        size_kb = (DATA_DIR / name).stat().st_size / 1024
        print(f"  data/{name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
