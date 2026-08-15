# Explicit Entanglement v1 — Hardware Results

**Date:** 2026-08-15  
**Job ID:** da09v7fo3ppc73al8ggg  
**Backend:** ibm_kingston (active)  
**Circuit:** `prime_gaps_entangled.py` — Bell pair + RY gap encoding + inverse QFT  
**Transpilation:** optimization_level=3, seed_transpiler=42  
**Shots:** 8,192

---

## Circuit

```
     ┌───┐   Bell pair  ┌──────────┐  gaps [1,2]  ┌─────────┐  iQFT  ┌─┐
q_0: ┤ H ├──■─────░─────┤ Ry(π/14) ├──────░────────┤0        ├───░────┤M├
     └───┘┌─┴─┐   ░     ├─────────┬┘      ░        │  qft_dg │   ░    └╥┘
q_1: ─────┤ X ├───░─────┤ Ry(π/7) ├───────░────────┤1        ├───░─────╫─┤M├
          └───┘   ░     └─────────┘       ░        └─────────┘   ░     ║  └╥┘
```

**Encoding:** First 2 prime gaps [1, 2] mapped to RY angles [π/14, π/7]  
(proportional to gap / max_gap, where max_gap = 14)

**Transpiled circuit (Kingston-class, opt=3):** depth=18, ECR gates=3, total gates=24

---

## Results

| Metric | Simulator | Hardware | Δ |
|--------|-----------|----------|---|
| Shots | 8,192 | 8,192 | — |
| \|00⟩ | 4,011 (48.96%) | 3,988 (48.68%) | −0.28pp |
| \|01⟩ | 2,102 (25.66%) | 2,003 (24.45%) | −1.21pp |
| \|10⟩ | 46 (0.56%) | 122 (1.49%) | +0.93pp |
| \|11⟩ | 2,033 (24.82%) | 2,079 (25.38%) | +0.56pp |
| Pearson r (q0–q1) | +0.552 | +0.541 | −0.011 |
| MI corrected (bits) | 0.2681 | 0.2446 | −0.0235 |
| Null MI mean | ~0.000 | ~0.000 | — |
| z-score vs null | 2,313 ★ | 2,398 ★ | — |

★ = significant (z > 2)

---

## Key findings

### Entanglement survives hardware

The Bell-pair structure holds on real hardware. MI drops from 0.268 bits (sim) to 0.245 bits (hardware) — a 9% reduction, entirely consistent with decoherence on a depth-18 circuit. Both remain massively significant (z > 2000). The suppressed |10⟩ outcome is the clearest fingerprint of entanglement: it appears at 0.56% in simulation and rises to only 1.49% on hardware, staying far below the 25% you'd expect from independent qubits.

### Hardware vs. StatePrep baseline

Compare to the previous 7-qubit StatePrep + inverse QFT run (archive/2026-08-15):

| | Old (StatePrep, 7q) | New (Bell+RY, 2q) |
|-|---|---|
| Root MI (sim) | 0.860 bits | 0.268 bits |
| Root MI (hw) | 0.060 bits | 0.245 bits |
| HW/Sim ratio | 7% | **91%** |
| Circuit depth | deep (StatePrep) | 18 |
| 2q gates | many | 3 |

The old circuit lost 93% of its entanglement signal on hardware. This circuit retains 91%. The explicit Bell-pair + minimal gate count is directly responsible — fewer gates means less decoherence exposure.

### Distribution structure

The outcome distribution is highly non-uniform:
- |00⟩ and |11⟩ together capture ~74% of shots (correlated outcomes)
- |10⟩ is strongly suppressed (~1.5% on hardware vs. 25% for independent qubits)
- This asymmetry is the iQFT's interference pattern acting on the Bell-entangled + gap-encoded state

The hardware distribution matches simulation closely. The main deviation is |10⟩ rising from 0.56% to 1.49% — small in absolute terms and consistent with single-qubit readout error on ibm_kingston.

---

## What's next

1. **Scale to 4 qubits** — encode gaps [1, 2, 2, 4] with the same Bell+RY+iQFT structure, compare MI retention ratio
2. **Tweak the encoding** — current angles use gap/max_gap normalisation over all 49 gaps (max=14), making small gaps very small rotations. Try normalising to the local window instead
3. **Noisy simulation** — add Kingston's noise model to AerSimulator to predict hardware MI before submitting, tightening the sim→hw calibration loop
4. **Approximated iQFT at n≥3** — the approximation_degree parameter starts pruning gates at n=3+; test whether MI is preserved as the iQFT is truncated
