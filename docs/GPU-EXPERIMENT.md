# GPU experiment brief — `xruntime-conformance`

**Cross-runtime conformance at scale**

This is the slice of the master queue that concerns this repository. The full document,
with all ten experiments and their decision trees, is at
[`SuperInstance/fleet-triage` → `docs/GPU-EXPERIMENTS.md`](https://github.com/SuperInstance/fleet-triage/blob/main/docs/GPU-EXPERIMENTS.md).

1. **State whether CUDA or CPU actually ran.** A CPU fallback reported as a GPU result is a
   fabricated measurement. Record the device string with the number.
2. **Report the variance, not just the mean.** If the measured quantity has `std == 0` over
   the sample, the result is **INCONCLUSIVE, never PASSED**. This is a hard rule: an earlier
   sharding experiment scored `PASSED` on the literal comparison `0 < 0`.
3. **The ground truth must be computed, not asserted.** Every game number below is exact.
   If a label is approximate, say by how much.
4. **Non-degeneracy is a precondition, not a result.** Assert that the data has variance
   *before* evaluating any relational claim about it.
5. **A control must vary the thing it audits, by a different path than the audited thing.**
   The most expensive mistake in this project's history: a control built from the same call
   path as the probes, which therefore confirmed a fault instead of auditing it.
6. **A ratio whose denominator can be zero is a construction, not a measurement.** It always
   produces a finding, and the finding is always false.
7. **Seed everything and record the seed.** Two runs of the same experiment that differ are
   a finding about the seed, not about the method.
8. **Cross-port arithmetic must preserve the reference loop/summation order.** Float addition
   is not associative; a port that vectorises changes the answer.

---

### Experiment 8 — Cross-runtime conformance at scale

**Repo:** `xruntime-conformance`. FNV-1a 64 of `"café Δ 日本語"` = `0x24a555471370b18d`, the
substrate-wide canary. **BLAKE2b/SHA-256 is for integrity; FNV-1a 64 is for cross-port
conformance.** They are not interchangeable and a run that conflates them is not a test.

A real portability bug already lives here: the `scalarSha`/`lossSha` mismatch across runtimes,
caused by summation order.

| Result | What it means |
|---|---|
| All ports agree bit-for-bit on a wide sweep | Conformance holds at scale. Then the canary is worth promoting to a release gate. |
| Divergence appears only above a size threshold | A float-accumulation-order bug. Fix the loop order; **do not loosen the tolerance**, that hides the class. |
| Divergence is random per-run | Non-determinism (atomics, threading), not arithmetic. That is a different defect with a different fix. |

---

## 4. Reporting format

Every experiment reports, in this order:

1. **Device.** Did CUDA actually run? Paste the device string.
2. **Data provenance.** Which commit, which digest, how many states/positions, and the
   FNV-1a 64 of the input file.
3. **The ceiling.** What is perfect, and what did you get as a fraction of it.
4. **Variance.** Mean ± std over N seeds, with the seeds listed. **std == 0 means INCONCLUSIVE.**
5. **The branch.** Which row of which decision tree above you landed on, quoted.
6. **Controls.** What ran that could have failed. If nothing could have failed, say that —
   it is the finding.
