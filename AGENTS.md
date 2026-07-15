# AGENTS.md

## Cursor Cloud specific instructions

DeepFilterNet is a Rust + Python monorepo for real-time speech enhancement (noise
suppression) of 48 kHz audio. There are **no servers, ports, or databases** — the product
ships as CLI tools, a Python framework, and audio plugins. The two core deliverables to
exercise the product end to end are:

1. `deep-filter` — Rust CLI that denoises `.wav` files (no Python needed).
2. `deepFilter` / `df/enhance.py` — Python framework (PyTorch) for enhancement/eval/training.

Standard commands live in `README.md`, the per-crate `Cargo.toml`s, `DeepFilterNet/pyproject.toml`
(poetry `poe` tasks), and the CI workflows under `.github/workflows/` (`test_df.yml`,
`rust_lint.yml`, `python_lint.yml`). Prefer those as the source of truth.

### Environment specifics (modernized stack)

- **Python 3.10–3.13 are supported** (validated on 3.12). 3.8/3.9 still work but have no
  NumPy 2 wheels, so they stay on NumPy 1.x.
- **NumPy 1.x and 2.x are both supported at runtime.** On Python ≥3.10 the default (and the
  `poetry.lock` pin) is **NumPy 2.x**; on 3.8/3.9 it is NumPy 1.x. This split is encoded via
  markers in `DeepFilterNet/pyproject.toml`, so `poetry install` stays reproducible across
  the whole range. If you edit dependency constraints, re-run `poetry lock` and keep
  `poetry check --lock` green.
- **PyTorch ≥2.4** (installed via the `poe install-torch-*` tasks, CPU/CUDA index URLs).
- The Python package is installed with `--no-root`; set `PYTHONPATH=$PWD/DeepFilterNet`
  (repo root: `PYTHONPATH=/workspace/DeepFilterNet`) when running `df/*.py` directly, or use
  `poetry run`.

### System dependencies

- `libhdf5-dev` — required to build `deep_filter` with `--all-features`/`dataset` and the
  `pyDF-data` dataloader (training). Not needed for inference.
- `libasound2-dev` + `pkg-config` — required to build the `df-demo` desktop app.
- The `df-demo` GUI and `cargo fmt`/`clippy` use the **pinned nightly** toolchain
  (`nightly-2026-07-15`, see `rust_lint.yml`); the GUI also needs a display.

### Running the product (from repo root)

- Rust CLI (build once, then run):
  `cargo run -p deep_filter --profile=release --features=tract,bin,wav-utils,transforms --bin deep-filter -- ./assets/noisy_snr0.wav -m ./models/DeepFilterNet3_onnx.tar.gz -o out_DeepFilterNet3`
- Python enhancement:
  `cd DeepFilterNet && PYTHONPATH=$PWD poetry run deepFilter --no-delay-compensation ../assets/noisy_snr0.wav -m DeepFilterNet3 -o ../out_py`
  (pretrained models auto-download to `~/.cache/DeepFilterNet`).

### Building the native modules

- `libdf` (inference bindings): `cd pyDF && maturin develop --release` → builds a single
  `cp38-abi3` wheel that imports on CPython 3.8+.
- `libdfdata` (training dataloader, Linux): `cd pyDF-data && maturin develop --release`
  (needs `libhdf5-dev`).

### Tests / lint

- Python model tests: `cd DeepFilterNet && poetry run python df/scripts/test_df.py`
  (needs the `eval` extra: pesq/pystoi/scipy, plus dnsmos utils).
- Rust tests: `cargo test --all-features -p deep_filter` (needs `libhdf5-dev`).
- Rust lint: `cargo +nightly-2026-07-15 fmt --all -- --check` and
  `cargo +nightly-2026-07-15 clippy -p <crate> --tests -- -D warnings`.
- Python lint (CI installs these standalone, not via poetry): `flake8`,
  `black --check --diff .`, `isort . --check --diff`.
- Micro-benchmark of the Rust↔Python boundary: `python scripts/bench_libdf.py` (needs only
  `numpy` + the `libdf` wheel).
