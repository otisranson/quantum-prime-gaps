"""Hierarchical qubit-correlation analysis of quantum_prime_gaps.py's 7-qubit
prediction-circuit hardware/simulator run.

Unlike quantum_radio (a shallow Hadamard + nearest-neighbor-CX-chain circuit,
where a hierarchical split is arbitrary by construction), this circuit is
genuinely globally entangling: it amplitude-encodes the prime-gap sequence via
`QuantumCircuit.initialize` (a StatePreparation instruction over the full
register -- see quantum_prime_gaps.py's `build_amplitude_circuit`), then applies
an inverse QFT across all 7 qubits, which mixes every qubit with every other by
construction. There's still no "encoding subspace vs. ancilla" split here -- all
7 qubits jointly carry the amplitude-encoded data -- so no highlighting is drawn
on these plots; the qubit ordering has no special structure to mark. But unlike
quantum_radio, real, strong correlation across the whole register is exactly what
this circuit should produce, which makes it a useful contrast case for the same
methodology.

The statistical core (marginals, correlation, bias-corrected/null-calibrated
mutual information, plotting) lives in ../qubit_hierarchy_core.py, shared with
quantum_radio/qubit_hierarchy_analysis.py.

Reads quantum_prime_gaps_results_7q_prediction.json (bitstring -> shot count, for
the real ibm_kingston hardware run -- job d9tso90u5hac73agdrk0, the same job
`fetch_first_hardware_run` re-fetches for the DD comparison in
quantum_prime_gaps.py itself -- and an AerSimulator run of the *identical*
circuit at the same shot count). That JSON is checked into the repo; `regenerate_data()`
below rebuilds it from scratch if it's ever missing. Produces the same four
analyses as quantum_radio's version: marginals, pairwise correlation matrix
with 2-sigma divergence flagging, a hierarchical MI dendrogram, and a written report.

Run: .venv/bin/python quantum_prime_gaps/qubit_hierarchy_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import qubit_hierarchy_core as core  # noqa: E402

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "quantum_prime_gaps_results_7q_prediction.json"
SCREENSHOTS_DIR = HERE / "screenshots"
REPORT_PATH = HERE / "qubit_hierarchy_report.md"

N_QUBITS = 7
HW_LABEL = "Hardware (ibm_kingston)"


def regenerate_data() -> None:
    """Rebuild RESULTS_PATH from scratch: the exact prediction circuit
    quantum_prime_gaps.py's --hardware pathway builds (amplitude-encoded prime
    gaps via StatePreparation, then inverse QFT), re-fetch its real hardware
    counts (job d9tso90u5hac73agdrk0 -- a free, read-only API call, the same one
    `fetch_first_hardware_run` uses, no quota cost), and run the identical
    circuit through AerSimulator at the same shot count so the two datasets are
    apples-to-apples. Only needed once; RESULTS_PATH is checked into the repo."""
    sys.path.insert(0, str(HERE))
    from qiskit import transpile  # noqa: E402
    from qiskit.circuit.library import QFTGate  # noqa: E402
    from qiskit_aer import AerSimulator  # noqa: E402
    from qiskit_ibm_runtime import QiskitRuntimeService  # noqa: E402

    from quantum_prime_gaps import FIRST_50_PRIMES, amplitude_encode, build_amplitude_circuit, prime_gaps  # noqa: E402

    job_id = "d9tso90u5hac73agdrk0"
    gaps = prime_gaps(FIRST_50_PRIMES)
    normalized, _norm = amplitude_encode(gaps, N_QUBITS)
    circuit = build_amplitude_circuit(normalized)
    circuit.append(QFTGate(N_QUBITS).inverse(), range(N_QUBITS))

    service = QiskitRuntimeService()
    job = service.job(job_id)
    result = job.result()
    hardware_counts = {k: int(v) for k, v in dict(result[0].data.meas.get_counts()).items()}
    shots = sum(hardware_counts.values())
    backend_name = job.backend().name

    measured = circuit.copy()
    measured.measure_all()
    simulator = AerSimulator()
    transpiled = transpile(measured, backend=simulator, optimization_level=1)
    simulator_counts = {k: int(v) for k, v in simulator.run(transpiled, shots=shots).result().get_counts().items()}

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "n_qubits": N_QUBITS,
                "shots": shots,
                "circuit": "amplitude-encoded prime gaps (StatePreparation) + inverse QFT -- the prediction-circuit pathway",
                "hardware_backend": backend_name,
                "hardware_job_id": job_id,
                "simulator_counts": simulator_counts,
                "hardware_counts": hardware_counts,
            },
            indent=2,
        )
    )


def load_counts() -> tuple[dict[str, int], dict[str, int], int, str]:
    data = json.loads(RESULTS_PATH.read_text())
    assert data["n_qubits"] == N_QUBITS, f"expected a {N_QUBITS}-qubit run, got {data['n_qubits']}"
    return data["simulator_counts"], data["hardware_counts"], data["shots"], data["hardware_backend"]


def write_report(
    sim_p1: np.ndarray,
    hw_p1: np.ndarray,
    flagged: list[tuple[int, int, float, float, float]],
    threshold: float,
    n_shots: int,
    backend_name: str,
    marginals_path: Path,
    correlations_path: Path,
    dendrogram_path: Path,
    sim_tree: dict,
    hw_tree: dict,
    significant: list[str],
) -> None:
    lines = [
        "# Hierarchical qubit analysis -- quantum_prime_gaps 7-qubit prediction circuit",
        "",
        "Same methodology as quantum_radio's hierarchical qubit analysis, applied to",
        "quantum_prime_gaps.py's real 7-qubit prediction-circuit run: amplitude-encoded",
        "prime gaps (`QuantumCircuit.initialize` -- a StatePreparation instruction over",
        "the full register) followed by an inverse QFT across all 7 qubits. Unlike",
        "quantum_radio's circuit, **the QFT genuinely mixes every qubit with every",
        "other by construction** -- there's no arbitrary bipartition to draw here (all 7",
        "qubits jointly carry the encoded data, no separate ancilla register), but real,",
        "strong correlation across the whole register is exactly what this circuit",
        "should produce, unlike quantum_radio's near-null result.",
        "",
        f"Hardware: job `d9tso90u5hac73agdrk0` on **{backend_name}**, {n_shots} shots (the same job",
        "`quantum_prime_gaps.py --dynamical-decoupling` re-fetches as \"first hardware run\").",
        "Simulator: AerSimulator run of the identical circuit at the same shot count, so both",
        "datasets carry comparable finite-sample noise.",
        "",
        f"Generated by `qubit_hierarchy_analysis.py`. Source: `{RESULTS_PATH.name}`.",
        "",
        "## Marginals",
        "",
        f"![marginals]({marginals_path.relative_to(HERE)})",
        "",
        "| Qubit | P(=1) sim | P(=1) hw | Δ |",
        "|---|---:|---:|---:|",
    ]
    for i in range(N_QUBITS):
        lines.append(f"| q{i} | {sim_p1[i]:.4f} | {hw_p1[i]:.4f} | {hw_p1[i] - sim_p1[i]:+.4f} |")

    lines += [
        "",
        "## Pairwise correlations",
        "",
        f"![correlations]({correlations_path.relative_to(HERE)})",
        "",
        "## Hierarchical partition tree",
        "",
        "Edge weights are Miller-Madow bias-corrected mutual information (bits) between",
        "each node's two halves, calibrated against a simulated independence null (200",
        "synthetic independent-qubit datasets per node, same marginals and shot count --",
        "see quantum_radio's report for why this matters more than the raw MI number).",
        f"Root-level MI (q{sim_tree['left']['qubits'][0]}-q{sim_tree['left']['qubits'][-1]} vs "
        f"q{sim_tree['right']['qubits'][0]}-q{sim_tree['right']['qubits'][-1]}):",
        f"simulator={sim_tree['mi']:.4f} bits (z={sim_tree['z']:.2f}), hardware={hw_tree['mi']:.4f} bits",
        f"(z={hw_tree['z']:.2f}).",
        "",
        f"![dendrogram]({dendrogram_path.relative_to(HERE)})",
        "",
        "**Partition-tree nodes exceeding z > 2 (simulator or hardware):**",
        "",
    ]
    if not significant:
        lines.append(
            "None. Somewhat surprising given the QFT's global mixing -- see the discussion in "
            "the console output / commit message for why this doesn't contradict the correlation "
            "matrix below."
        )
    else:
        lines.extend(f"- {line}" for line in significant)
    lines += [
        "",
        "## Qubit pairs flagged beyond 2σ",
        "",
        f"2σ threshold on |hw correlation - sim correlation|, from the {n_shots}-shot sampling",
        f"noise on each Pearson estimate: **{threshold:.4f}**.",
        "",
    ]
    if not flagged:
        lines.append("None -- every pairwise correlation's hardware/simulator difference is within sampling noise.")
    else:
        lines.append("| Pair | sim corr | hw corr | Δ |")
        lines.append("|---|---:|---:|---:|")
        for i, j, sc, hc, delta in flagged:
            lines.append(f"| q{i}-q{j} | {sc:+.4f} | {hc:+.4f} | {delta:+.4f} |")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines))


def main() -> None:
    if not RESULTS_PATH.exists():
        regenerate_data()
    sim_counts, hw_counts, n_shots, backend_name = load_counts()
    sim_bits, sim_probs = core.counts_to_bits_and_probs(sim_counts, N_QUBITS)
    hw_bits, hw_probs = core.counts_to_bits_and_probs(hw_counts, N_QUBITS)

    sim_p1 = core.marginals(sim_bits, sim_probs)
    hw_p1 = core.marginals(hw_bits, hw_probs)
    marginals_path = core.plot_marginals(
        sim_p1,
        hw_p1,
        N_QUBITS,
        SCREENSHOTS_DIR / "qubit_hierarchy_marginals.png",
        highlight_upto=None,
        hw_label=HW_LABEL,
        title_note="\n(amplitude-encoded prime gaps + inverse QFT -- no encoding/ancilla split to highlight)",
    )

    sim_corr = core.correlation_matrix(sim_bits, sim_probs)
    hw_corr = core.correlation_matrix(hw_bits, hw_probs)
    flagged = core.flag_divergent_pairs(sim_corr, hw_corr, n_shots, N_QUBITS)
    threshold = 2 * np.sqrt(2.0 / (n_shots - 3))
    correlations_path = core.plot_correlations(
        sim_corr,
        hw_corr,
        [(i, j) for i, j, *_ in flagged],
        N_QUBITS,
        SCREENSHOTS_DIR / "qubit_hierarchy_correlations.png",
        highlight_upto=None,
        hw_label=HW_LABEL,
    )

    rng = np.random.default_rng(0)
    sim_tree = core.build_partition_tree(sim_bits, sim_probs, list(range(N_QUBITS)), n_shots, sim_p1, rng)
    hw_tree = core.build_partition_tree(hw_bits, hw_probs, list(range(N_QUBITS)), n_shots, hw_p1, rng)
    dendrogram_path = core.plot_dendrograms(
        sim_tree,
        hw_tree,
        N_QUBITS,
        SCREENSHOTS_DIR / "qubit_hierarchy_dendrogram.png",
        highlight_upto=None,
        hw_label=HW_LABEL,
        title_note=" No highlighted subspace -- the QFT mixes the whole register.",
    )

    sig: list[str] = []
    core.significant_nodes(sim_tree, "simulator", sig)
    core.significant_nodes(hw_tree, "hardware", sig)

    write_report(
        sim_p1,
        hw_p1,
        flagged,
        threshold,
        n_shots,
        backend_name,
        marginals_path,
        correlations_path,
        dendrogram_path,
        sim_tree,
        hw_tree,
        sig,
    )

    print(f"Backend: {backend_name}, {n_shots} shots")
    print(f"Root-level MI: simulator={sim_tree['mi']:.4f} bits (z={sim_tree['z']:.2f}), hardware={hw_tree['mi']:.4f} bits (z={hw_tree['z']:.2f})")
    print(f"2-sigma correlation-divergence threshold: {threshold:.4f}")
    print(f"Flagged pairs: {len(flagged)} / {N_QUBITS * (N_QUBITS - 1) // 2}")
    for i, j, sc, hc, delta in flagged:
        print(f"  q{i}-q{j}: sim={sc:+.4f} hw={hc:+.4f} delta={delta:+.4f}")
    print(f"Partition-tree nodes exceeding the z>2 independence-null significance bar: {len(sig)}")
    for line in sig:
        print(f"  {line}")
    print(f"\nWrote {marginals_path}, {correlations_path}, {dendrogram_path}, {REPORT_PATH}")


if __name__ == "__main__":
    main()
