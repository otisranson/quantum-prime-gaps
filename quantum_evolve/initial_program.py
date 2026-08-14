# Copyright 2026 Otis Ranson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Seed program for OpenEvolve: the frequency-truncation-and-reconstruction step
shared by quantum_prime_gaps.py's classical and quantum-circuit prime-gap
forecasts.

quantum_prime_gaps.py predicts primes 41-50 from primes 1-40 two ways: a
classical np.fft.fft spectrum, and a real Qiskit QFTGate spectrum computed on a
zero-padded amplitude-encoded register (`quantum_fft`, unchanged here -- that
part is separately verified against np.fft.fft to floating-point precision and
isn't what's being evolved). Both pathways feed their spectrum through the same
final step: keep the `top_k` strongest frequency components, zero the rest, and
evaluate the continuous-time inverse DFT past the known window. That step --
`reconstruct` below -- is copied verbatim from `_dft_reconstruct` in
quantum_prime_gaps.py, and is what's inside the EVOLVE-BLOCK.

Documented baseline (see quantum_prime_gaps/quantum_prime_gaps.py's module
docstring and README): on backward verification, neither pathway currently
beats two naive baselines (repeat the mean known gap; repeat the last known
gap) at any top_k. evaluator.py scores candidates on exactly this benchmark --
try to actually beat the baselines, on both pathways, without overfitting to
just one of them.

Hard constraint enforced by the evaluator, not just this docstring: called with
top_k=None, reconstruct(spectrum, n, arange(n), None) must exactly reproduce
the known values it was built from (spectrum = fft(values)) -- that's the
definition of being a correct inverse DFT, not an optional nicety.
"""

# EVOLVE-BLOCK-START
import numpy as np


def frequency_components(n: int) -> list[tuple[int, ...]]:
    """Bin index -> the set of bin indices that must be kept or dropped together
    for the reconstruction to stay real-valued (a bin and its conjugate mirror)."""
    components = []
    for k in range(n // 2 + 1):
        mirror = (-k) % n
        components.append((k,) if mirror == k else (k, mirror))
    return components


def ranked_frequency_components(spectrum: np.ndarray, n: int) -> list[tuple[tuple[int, ...], float]]:
    """Frequency-bin groups ranked by combined magnitude, strongest first."""
    components = frequency_components(n)
    ranked = [(bins, float(np.sum(np.abs(spectrum[list(bins)])))) for bins in components]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def reconstruct(spectrum: np.ndarray, n: int, t_values: np.ndarray, top_k: int | None = None) -> np.ndarray:
    """Evaluate the inverse-DFT sum x(t) = (1/n) * sum_k X_k * exp(2j*pi*k*t/n) at
    arbitrary times t_values, including t >= n -- the step that reads the frequency
    domain as a continuous function and forecasts past the known window.

    If top_k is given, every frequency component except the top_k strongest
    (by combined magnitude, mirrors kept together) is zeroed before evaluating.
    """
    if top_k is not None:
        ranked = ranked_frequency_components(spectrum, n)
        keep_bins = {b for bins, _ in ranked[:top_k] for b in bins}
        mask = np.zeros(n, dtype=bool)
        mask[list(keep_bins)] = True
        spectrum = np.where(mask, spectrum, 0)

    t = np.asarray(t_values, dtype=float)
    k = np.arange(n)
    phase = 2j * np.pi * np.outer(t, k) / n
    return (np.exp(phase) @ spectrum).real / n


# EVOLVE-BLOCK-END
