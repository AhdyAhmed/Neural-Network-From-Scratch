# Day 13 — Experiments & Results

## Goal

Make experiments reproducible, compare a small set of controlled alternatives, and report only measured values.

## Run it

From the repository root:

```bash
python examples/day13_experiments.py
```

The first run downloads MNIST through the existing checksum-verified loader. It trains the predeclared final configuration, runs four validation-only ablations, evaluates the held-out test set only for the final configuration, and writes:

- `results/day13_experiments.json` — configs, seeds, split sizes, measured metrics, per-epoch history, and elapsed time.
- `assets/day13_experiments.png` — final-model learning curves and ablation validation accuracy.

A small deterministic smoke run is available with `python examples/day13_experiments.py --quick`. It is for checking the pipeline only: it uses subsets, trains for two epochs, and deliberately does not report test accuracy. Use `--ablation` to run only ablations, `--no-ablation` to run only the final model, and `--train-limit`, `--val-limit`, or `--test-limit` to cap split sizes.

## Final configuration

| Setting | Value |
|---|---|
| Architecture | 784 → 128 → 64 → 10 |
| Hidden activation | ReLU |
| Output | Softmax |
| Loss / optimizer | Categorical cross-entropy / Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Maximum epochs | 15 |
| Hidden-layer initialization | He normal |
| Dropout | 0.10 after each hidden activation |
| Seed | 2026 (shuffle seed 2027) |
| Early stopping | Validation loss, patience 3, min_delta 0.0001 |

The output layer uses Xavier initialization. The final model's held-out test metrics are evaluated once after training; ablation choices are compared on validation metrics only. In quick mode, the test metric is omitted rather than presented as a comparable benchmark.

## Ablations

The script compares a baseline without dropout, adding 0.10 dropout, switching hidden-layer initialization from He to Xavier, and reducing Adam's learning rate from 0.001 to 0.0003. These are small controlled experiments, not a broad hyperparameter search. Each run records its full configuration and deterministic seeds.

## Interpreting results honestly

Use the numbers written by your own run in `results/day13_experiments.json`; don't copy numbers from another machine as if they were yours. Training time depends on CPU, NumPy/BLAS, and machine load. A small validation difference on one seed is not proof that a setting is universally better. For stronger conclusions, repeat promising configurations across multiple seeds and report mean and standard deviation.
