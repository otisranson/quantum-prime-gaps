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

"""Evaluator for quantum_evolve: scores a candidate `reconstruct` function
against quantum_prime_gaps.py's own backward-verification benchmark (predict
primes 41-50's gaps from primes 1-40's), for both the classical (np.fft.fft)
and quantum-circuit (quantum_fft) spectrum pathways, at several truncation
budgets (top_k). See initial_program.py's docstring for the full setup.

quantum_fft runs an exact Qiskit Statevector simulation (no hardware, no
shots), so a full evaluation is cheap -- this never touches IBM Quantum.
"""

import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np
from openevolve.evaluation_result import EvaluationResult

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "quantum_prime_gaps"))
import quantum_prime_gaps as qpg  # noqa: E402

N_QUBITS = 7
N_PREDICT = 10
TOP_KS = [1, 2, 3, 5, 8]


def _load_reconstruct(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "reconstruct"):
        raise AttributeError("candidate program has no 'reconstruct' function")
    return module.reconstruct


def _backward_setup() -> tuple[np.ndarray, np.ndarray, float]:
    """Mirrors quantum_prime_gaps.py's own _backward_split: first 39 known gaps,
    the 10 held-out actual gaps, and the better of its two naive baselines."""
    all_gaps = qpg.prime_gaps(qpg.FIRST_50_PRIMES)
    known_gaps = all_gaps[:39]
    actual_gaps = all_gaps[39 : 39 + N_PREDICT]
    baseline_mean_mae = float(np.mean(np.abs(known_gaps.mean() - actual_gaps)))
    baseline_last_repeat_mae = float(np.mean(np.abs(known_gaps[-1] - actual_gaps)))
    return known_gaps, actual_gaps, min(baseline_mean_mae, baseline_last_repeat_mae)


def _fail(message: str, extra_artifacts: dict | None = None) -> EvaluationResult:
    return EvaluationResult(
        metrics={"combined_score": 0.0, "error": message},
        artifacts={"traceback": traceback.format_exc(), **(extra_artifacts or {})},
    )


def evaluate(program_path: str) -> EvaluationResult:
    known_gaps, actual_gaps, baseline = _backward_setup()

    try:
        reconstruct = _load_reconstruct(program_path)
    except Exception as e:
        return _fail(f"failed to load candidate: {e}")

    n = len(known_gaps)
    classical_spectrum = np.fft.fft(known_gaps)

    # Hard correctness gate, mirroring quantum_prime_gaps.py's own
    # verify_extrapolation_roundtrip: a full-spectrum (top_k=None) reconstruction
    # evaluated at the known sample points must exactly reproduce them. A
    # candidate that breaks this isn't a valid inverse DFT, whatever its
    # backward-verification MAE looks like -- score it zero before even
    # computing one.
    try:
        roundtrip = reconstruct(classical_spectrum, n, np.arange(n), None)
    except Exception as e:
        return _fail(f"roundtrip call crashed: {e}")
    if not np.allclose(roundtrip, known_gaps, atol=1e-6):
        return EvaluationResult(
            metrics={"combined_score": 0.0, "roundtrip_ok": 0.0, "error": "broke the inverse-DFT roundtrip identity"},
            artifacts={"suggestion": "reconstruct(spectrum, n, arange(n), top_k=None) must reproduce the known values exactly"},
        )

    quantum_spectrum = qpg.quantum_fft(known_gaps, N_QUBITS)
    dim = 2**N_QUBITS
    future_t = np.arange(n, n + N_PREDICT)

    margins = {}
    try:
        for top_k in TOP_KS:
            classical_forecast = reconstruct(classical_spectrum, n, future_t, top_k)
            margins[f"classical_top{top_k}"] = baseline - float(np.mean(np.abs(classical_forecast - actual_gaps)))

            quantum_forecast = reconstruct(quantum_spectrum, dim, future_t, top_k)
            margins[f"quantum_top{top_k}"] = baseline - float(np.mean(np.abs(quantum_forecast - actual_gaps)))
    except Exception as e:
        return _fail(f"forecast crashed: {e}")

    if not all(np.isfinite(v) for v in margins.values()):
        return _fail("non-finite forecast (NaN/Inf) somewhere in the top_k sweep")

    avg_margin = float(np.mean(list(margins.values())))
    worst_margin = float(np.min(list(margins.values())))
    # Positive margin means beating the baseline. Score is split evenly between
    # the average margin and the worst single (pathway, top_k) margin, then
    # squashed to (0, 1), so a candidate can't win by overfitting one condition
    # while quietly getting worse everywhere else.
    combined_score = float(1.0 / (1.0 + np.exp(-(0.5 * avg_margin + 0.5 * worst_margin))))

    metrics = {
        "combined_score": combined_score,
        "avg_margin": avg_margin,
        "worst_margin": worst_margin,
        "roundtrip_ok": 1.0,
        **margins,
    }
    return EvaluationResult(metrics=metrics, artifacts={"baseline_mae": baseline})
