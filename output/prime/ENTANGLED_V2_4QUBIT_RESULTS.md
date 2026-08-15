# Explicit Entanglement v2 — 4-Qubit Hardware Results

**Date:** 2026-08-15  
**Job ID:** da0b8gt0vrcc73bq8lb0  
**Backend:** ibm_kingston (active)  
**Circuit:** `optimize_4qubit.py` — Bell pair + RY gap encoding + approx iQFT (degree=1)  
**Shots:** 8,192

---

## Circuit

**Architecture:** Bell pair seed (H→CNOT) + RY gap encoding on 4 qubits + approximated inverse QFT

**Gaps encoded:** [1, 2, 2, 4] (first 4 prime gaps: 3−2, 5−3, 7−5, 11−7)  
**RY angles:** [π/14, π/7, π/7, 2π/7] (gap × π / MAX_GAP, MAX_GAP=14 over all 49 gaps)

**iQFT variant:** `synth_qft_full(4, inverse=True, degree=1)` — removes the smallest CP gate (π/8 = 22.5°) per qubit, keeping 5 of 6 CP gates  
**Transpilation:** Real Kingston coupling map (352 edges), optimization_level=3, seed_transpiler=42  

---

## Depth Reduction Comparison

All strategies transpiled against Kingston's real coupling map (not FakeSherbrooke proxy). Baseline depth from prior run on FakeSherbrooke was 81; real-map transpilation already improves this to 69.

| Strategy | Depth | ECR | Sim Root MI | Pred Retention | Pred HW MI | Within 10%? |
|----------|------:|----:|------------:|---------------:|-----------:|:-----------:|
| Full iQFT (degree=0, baseline) | 69 | 19 | 0.3527 bits | 70.3% | 0.2480 | ✓ |
| **Approx iQFT degree=1** ← winner | **55** | **17** | **0.3489 bits** | **75.5%** | **0.2634** | **✓** |
| Approx iQFT degree=2 | 30 | 7 | 0.2542 bits | 85.8% | 0.2181 | ✗ |
| Approx iQFT degree=3 | 12 | 1 | 0.0000 bits | 94.1% | 0.0000 | ✗ |
| Linear iQFT (synth_qft_line) | 103 | 33 | 0.1208 bits | 59.1% | 0.0714 | ✗ |

**Tolerance criterion:** root MI ≥ 0.336 bits (within 10% of 0.373 bits baseline sim MI)

**Winner:** Approx degree=1 — best depth among MI-valid strategies. Removes one CP gate per qubit (CP(π/8)) while retaining the dominant phase structure. Reduces depth 81→55 (−32% vs FakeSherbrooke estimate, −20% vs true Kingston baseline).

**Why degree=2 fails the tolerance:** Removing CP(π/4) and CP(π/8) drops root MI to 0.254 (−32%), well below the 0.336 floor. The medium-angle CP gates carry significant inter-qubit correlation that can't be dropped without sacrificing the signal.

**Why linear iQFT is worse:** `synth_qft_line` targets nearest-neighbor topology but adds many SWAP gates that increase depth above the full-iQFT baseline. On 4 qubits, the standard synthesis + router does better.

---

## Hardware Results

| Metric | Simulator | Hardware | Δ |
|--------|----------:|----------:|---|
| Shots | 8,192 | 8,192 | — |
| ISA circuit depth | 55 (proxy) | 56 | +1 |
| Sim root MI | 0.3489 bits | — | — |
| Hardware root MI | — | 0.1839 bits | — |
| HW/Sim retention | — | **52.7%** | — |
| Predicted retention | — | 75.5% | −22.8pp |

**Top outcome counts (hardware):**

| State | Count | % |
|-------|------:|--:|
| \|0000⟩ | 2,499 | 30.5% |
| \|1100⟩ | 1,146 | 14.0% |
| \|0100⟩ | 1,061 | 13.0% |
| \|1010⟩ | 977 | 11.9% |
| \|0110⟩ | 842 | 10.3% |
| \|1000⟩ | 398 | 4.9% |
| \|0010⟩ | 336 | 4.1% |
| \|1110⟩ | 186 | 2.3% |

---

## Key Findings

### Retention vs prediction

Hardware retained 52.7% of simulation root MI — lower than the 75.5% prediction. The T2_depth model (calibrated at n=2, depth=18, 91% retention) underestimated decoherence at n=4. Likely sources:

1. **n=4 readout error compounds:** Four-qubit readout error is roughly 4× single-qubit error; each qubit at ~1% readout error compounds to a several-percent state-assignment error rate across the 4-bit register.
2. **Cross-qubit crosstalk:** A 4-qubit circuit spans a wider region of the heavy-hex chip, encountering more neighbor crosstalk than the 2-qubit control run used to calibrate T2_depth.
3. **SWAP overhead not captured by depth model:** The T2_depth exponential model treats all layers equally; SWAP-heavy layers have disproportionate decoherence because ECR gates have ~10× higher error than single-qubit gates.

### Entanglement signal present but attenuated

The hardware distribution is non-uniform and clearly structured — the top 5 outcomes capture 80% of shots. Random noise would give ~6.25% per state. The residual root MI (0.184 bits) is significantly above the noise floor, confirming that the Bell-pair entanglement structure partially survives at depth=56.

### Scaling lesson

| Circuit | Depth | ECR | Sim root MI | HW root MI | Retention |
|---------|------:|----:|------------:|-----------:|----------:|
| 2q Bell+RY (v1) | 18 | 3 | 0.268 bits | 0.245 bits | **91%** |
| 4q Bell+RY, approx-1 (v2) | 56 | 17 | 0.349 bits | 0.184 bits | **53%** |

Going from 2 to 4 qubits: depth triples, ECR count jumps 6×, and retention drops by 38 percentage points. The T2_depth model needs a 4q calibration point. Next step: run the full iQFT 4q variant (depth=69) on hardware to compare and bracket the model.

---

## What's Next

1. **Calibrate T2_depth at n=4** — run the full iQFT 4q circuit (depth=69) on Kingston; bracket the true retention curve between depth=56 (53%) and depth=69 (?%)
2. **Noisy simulation pre-flight** — add Kingston's error model to AerSimulator (`from qiskit_ibm_runtime import QiskitRuntimeService; noise_model = NoiseModel.from_backend(backend)`) to tighten the prediction loop before hardware submission
3. **Qubit selection** — pin to the best 4 connected qubits on Kingston (lowest T1/T2 decay, best readout fidelity) rather than letting Sabre choose; a better physical qubit chain could recover 10–15pp retention
4. **Encoding revisit** — small gaps (1, 2) map to small angles (π/14, π/7) under the global normaliser. Local normalisation (angle = gap × π / local_max) would spread angles more, potentially increasing baseline MI and making hardware losses less costly
