# Supplied candidate run

The README contains measured outputs from the candidate code, not illustrative
curves or hand-designed corruptions of healthy generated frames.

| Item | Recorded value |
| --- | --- |
| Date | 2026-11-05 |
| Execution | UBELIX Slurm job 21285250; completed, exit code 0:0 |
| Device | One NVIDIA GeForce RTX 4090 |
| Queue | `gpu-invest`, `job_gpu_preemptable`, account `gratis` |
| Training | 6,000 optimizer steps, batch size 64, AdamW, LR 0.0003 |
| Seed | 17 |
| Parameters | 255,873 |
| Candidate training wall time | 56.77 seconds, including initial CUDA setup |
| Peak PyTorch allocated device memory | 142,660,608 bytes |
| Training data | 256 episodes, 48 transitions each, collection seed 2026 |
| Separate diagnostic data | 32 episodes, 48 transitions each, seed 90000 |
| Sampling | 32 Euler steps per frame; horizon 12; noise seed 23 |
| Displayed trajectory | First episode in the diagnostic archive |

All training and data collection ran on a compute node. CPU checks do not train
the model. The saved environment uses Python 3.11.7 and PyTorch 2.8.0+cu126.

The model's validation loss uses the partition produced by `dataloader.py`.
The trajectory comparisons instead use the independent diagnostic archive, so
the two kinds of validation are not interchangeable. Training loss is averaged
over the most recent logging interval; validation averages over all validation
examples using fixed interpolation noise and times.

The MSE plots compare the evaluation entry point, the deployment generation entry
point and a persistence baseline. All start from the same context and use the
same actions. Evaluation and generation use identical seeds and batches. Images
are clamped to `[-1,1]` after each frame's ODE integration, then mapped to `[0,1]`
for metrics and display. The figures are a single-seed debugging case, not a
multi-seed benchmark.

Reproduction files:

- `assets/history.json`: every logged training and validation value.
- `assets/metrics.json`: split, data, action and trajectory probes.
- `assets/provenance.json`: source SHA-256 hashes, training archive hash, config,
  device and library versions.
- `assets/training_summary.json`: duration, parameter count and peak allocated memory.
- Release `v1.0-interview`, `candidate-run.zip`: candidate checkpoint, matching
  history/provenance/summary, and the complete installed environment freeze.

Regenerating a compressed archive may change its ZIP-level hash while leaving
all arrays identical. GPU/CPU floating-point behaviour and different dependency
versions can also cause small numerical differences. The bundled archives and
checkpoint are the exact objects used for the displayed diagnostics.
