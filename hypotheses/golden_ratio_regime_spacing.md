# Golden Ratio Regime Spacing — Circuit Resonance or Imprint?

**Date:** August 16, 2026

## Observation

The three known regime changepoints from the 5000-prime run are at windows 1529, 2501, and 4211. The ratio of the first two changepoints to the third:

- 2501 / 4211 = 0.594
- Golden ratio inverse 1/φ = 0.618

These are close but not exact.

**Correction before this went further:** I originally wrote that the golden ratio phase rotation φ = π·(1+√5)/2 is "already encoded in the quantum circuit itself" for this project. I checked, and that's not true here — `prime_predictor.py`, `prime_predictor_hw.py`, and `quantum_prime_gaps/` use only `RY` gap-encoding rotations, a Bell-pair `CNOT`, and QFT/iQFT (standard π/2^k phase gates). There is no `RZ` gate and no φ constant anywhere in this repo's circuit. The `RZ(π·φ)` rotation is real, but it belongs to a different circuit entirely: the sibling `quantum_radio` project (`QuantumResearch/quantum_radio/quantum_radio.py`), which deliberately uses φ as its phase-kick angle. I'm keeping that distinction explicit below because it changes what "Imprint" can mean for this project specifically.

## Hypothesis

I think the regime changepoints are landing at approximately golden ratio proportions of the total sequence length. This may indicate one of two things:

- **Resonance:** φ phase structure is genuinely sensitive to golden ratio structure latent in the prime gap sequence — a real mathematical relationship between φ and prime distribution.
- **Imprint:** A circuit is imposing its own phase signature onto the measurement statistics — the changepoints reflect a circuit's rotation angle, not the primes.

Given the correction above, "Imprint" can't mean this repo's own circuit imposing a φ signature — this repo's circuit doesn't contain φ anywhere, so it has no mechanism to imprint it. If regime spacing here really does track 1/φ, that's either coincidence at n=3 changepoints, or a genuine relationship between φ and prime gap structure independent of any circuit — not a self-inflicted artifact of this circuit's own design. That's a meaningfully different, and honestly less likely, claim than the original framing implied.

## Critical test

Run the same regime change analysis using a circuit with random phase rotations, or a non-φ constant. If the changepoints still land at golden ratio proportions — resonance. If they shift — imprint. Given the correction above, this test matters even more than originally framed: this repo's circuit has no φ built in, so if it's not resonance, "imprint" would have to come from somewhere else in the pipeline (e.g. the iQFT's own fixed phase angles) rather than from a deliberately chosen φ constant — worth checking for before assuming resonance.

## Connection to Quantum Radio

The TVD of 0.3914 between hardware and simulation (`QuantumResearch/quantum_radio/quantum_radio_report_12q.md`, 12-qubit run) is from the sibling `quantum_radio` project, which does use `RZ(π·φ)` as a deliberate phase-kick. I think this divergence may partly reflect φ structure diverging under real quantum noise — hardware and simulation may be processing the φ rotation differently at scale. This is a separate question from the regime-spacing observation above, since it's about a different circuit and a different project; any connection between the two is speculative until the critical test above is run.

## Correction

**Correction — August 16, 2026:**

The framing of this hypothesis contains a factual error. This repository's circuit does not use φ as a rotation angle. The φ rotation lives in the Quantum Radio repository, not here. The "imprint" mechanism described above does not apply to this circuit.

What remains:
- Three data points showing approximate φ proportions — this could be coincidence at n=3
- A question about whether φ structure is latent in prime gaps independent of any circuit design
- The control run is still the right test, but it's testing the prime gap structure claim, not a circuit imprint claim

The "highest priority" framing is premature given only three data points and a removed mechanism. Downgraded to: interesting observation, worth one control run, not yet a hypothesis with teeth.

## Status

Unverified. Control run not yet executed. Downgraded from "highest priority" per the Correction above — interesting observation, worth one control run, not yet a hypothesis with teeth.
