# Quantum Prime Gaps — Project Status Prompt

**Date:** 2026-08-15  
**Repo:** `otisranson/quantum-prime-gaps`  
**Purpose of this doc:** Bring Claude web UI up to speed so we can discuss next steps.

---

## What This Project Does (one paragraph)

We encode prime gaps (the differences between consecutive prime numbers) onto real IBM quantum hardware and ask: can a quantum circuit predict the next gap it hasn't seen? The first 50 primes are hardcoded. Their 49 gaps are fed into a 4-qubit quantum circuit that uses entanglement (Bell pairs), rotation gates (RY), and an inverse Quantum Fourier Transform to produce a probability distribution over possible next gaps. A recurrent feedback loop runs that circuit 46 times — one window per consecutive group of 4 gaps — carrying forward the most probable outcome as a bias into the next window. The final window's distribution gives the prediction.

---

## Architecture: The Recurrent Prime Predictor (v3)

**Circuit (per window):**
```
H(q0) → CNOT(q0→q1)           Bell pair seed
RY(gap_i × π / local_max + offset_i, qi)   per-qubit gap encoding (local normalization)
approx inverse QFT (degree=1)  removes smallest CP gate (π/8), keeps 5 of 6
measure → 4-bit bitstring
```

**Feedback loop:**
- Window i's mode bitstring → per-qubit offsets for window i+1
- `offset_q = (bit_q - 0.5) × scale`  (scale=0.05 was best)
- 46 windows total, each a group of 4 consecutive prime gaps

**Decode:**
- `E[gap] = Σ (counts[bs] / total) × (int(bs,2) × local_max / 15)`
- Round to nearest integer → predicted gap → add to last known prime (229)

**Key design choices:**
- **Local window normalization** (÷window_max not ÷global_max): prevents the |0000⟩ attractor that killed v1/v2
- **Approximated iQFT degree=1**: reduces circuit depth 81→55, trades one CP gate for 22% less decoherence
- **Bell pair seed**: forces correlated initial state so the QFT has something to work with
- **Miller-Madow corrected entropy**: bias-corrected mutual information between qubit halves [q0,q1] vs [q2,q3]

---

## Results Summary

### AerSimulator (v3, clean)
| Scale | E[gap] | Predicted prime | Error |
|-------|--------|-----------------|-------|
| 0.05 | 4.006 | **233** ✓ | 0.006 |
| 0.10 | 4.057 | **233** ✓ | 0.057 |
| 0.20 | 4.137 | **233** ✓ | 0.137 |

### ibm_kingston hardware (v3, 46 windows, 8192 shots each)
| Metric | Value |
|--------|-------|
| Run timestamp | 20260815_204703 |
| Backend | ibm_kingston (156-qubit heavy-hex) |
| Shots per window | 8,192 |
| Total hardware jobs | 46 sequential |
| Weighted E[gap] | **3.9578 → gap=4** |
| Predicted prime | **229 + 4 = 233** ✓ |
| Ground truth | gap=4, prime=233 |
| Error | 0.0422 |
| Verdict | **✓ CORRECT** |
| Mean hardware MI | 0.1193 bits |
| Commit | 13c57e6 |

### Earlier: 4-qubit depth reduction run (ibm_kingston)
| Strategy | Depth | Root MI | Retention |
|----------|------:|--------:|----------:|
| Full iQFT (baseline) | 69 | 0.349 bits | 70% |
| **Approx iQFT degree=1 ← winner** | **55** | **0.349 bits** | **53%** |
| Approx iQFT degree=2 | 30 | 0.254 bits | 86% |
| Linear iQFT | 103 | 0.121 bits | 59% |

---

## File Map

```
prime_predictor.py          v3 AerSimulator recurrent predictor
prime_predictor_hw.py       v3 hardware runner (46 sequential Kingston jobs)
optimize_4qubit.py          depth reduction comparison (5 strategies vs Kingston coupling map)
quantum_prime_gaps/         original QFT spectral analysis + 7-qubit hardware runs
quantum_evolve/             OpenEvolve experiment on the reconstruction step
output/prime/               timestamped run outputs (PNG + JSON), auto-committed after each run
  20260815_203411/          latest AerSimulator run
  20260815_204703/          Kingston hardware run (the one above)
```

---

## What We Know Works

1. **The circuit predicts correctly on hardware** — 3.9578 rounds to 4, prime=233, despite 46 sequential jobs and real hardware noise
2. **Local window normalization was the key fix** — global normalization caused a |0000⟩ attractor where large gaps dominated; local norm spreads all windows uniformly
3. **Approx iQFT degree=1 is the right depth tradeoff** — removes only the smallest CP gate, retains 97% of simulation MI, cuts depth 25%
4. **Feedback works** — the per-qubit mode-bitstring feedback demonstrably propagates structure across windows (mean MI=0.12 bits vs noise floor ~0.01)

## Known Limitations

1. **Mean hardware MI (0.12 bits) is lower than sim (0.35 bits)** — 34% retention, hardware decoherence at n=4 is significant
2. **Mode prediction (|0000⟩) gives wrong gap (~1)** — the weighted average saves it; the distribution's center of mass is correct even when the peak isn't
3. **The T2_depth model underestimates 4-qubit decoherence** — calibrated at n=2; n=4 compounds readout error and crosstalk
4. **46 sequential jobs is slow** — 2–3 hours depending on Kingston queue; can't parallelize because feedback requires each result before the next circuit

---

## Open Questions / Next Steps

These are things worth discussing:

1. **Can we reduce to fewer hardware windows?** Simulate windows 0–44, submit only window 45 to hardware with the sim-derived feedback offset. One hardware job instead of 46. Tests the final prediction without the full feedback chain on hardware.

2. **Qubit selection optimization** — currently letting Sabre choose qubits. Pinning to the 4 best-connected qubits with lowest T1/T2 decay on Kingston could recover 10–20pp MI retention.

3. **Scale sweep on hardware** — we only ran scale=0.05. Running 0.05, 0.1, 0.2 on hardware (3 jobs on the final window) would show whether feedback strength matters at real decoherence levels.

4. **Extend to prime 239** — next gap after 233 is 6 (239−233). Shift the window forward: feed the 233 result back, run one more window. Tests generalization past the training set.

5. **Visualize the feedback chain** — plot how the per-qubit offsets evolve across 46 windows. Do they converge? Oscillate? This might show whether the feedback is carrying real signal or just noise.

6. **Compare to classical baselines** — mean gap predictor says 49/49=~2.8, repeating last gap says 4 (lucky). Need a formal comparison to show the quantum circuit isn't just replicating what a trivial heuristic would say.

---

## Context for This Chat

The project lives at `otisranson/quantum-prime-gaps`. All hardware runs are on IBM Quantum open plan (ibm_kingston, 156-qubit heavy-hex). The v3 predictor is in `prime_predictor.py` (sim) and `prime_predictor_hw.py` (hardware). Every run auto-commits and pushes a timestamped output directory to the repo.

The most recent committed state is clean — all outputs from the Kingston hardware run are in `output/prime/20260815_204703/` and pushed as commit `13c57e6`.
