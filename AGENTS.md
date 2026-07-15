# AGENTS.md

## Cursor Cloud specific instructions

DeepFilterNet is a Rust + Python monorepo for real-time speech enhancement (noise
suppression) of 48 kHz audio. There are **no servers, ports, or databases** — the
product ships as CLI tools, a Python framework, and audio plugins. The two core
deliverables to exercise the product end to end are:

1. `deep-filter` — Rust CLI that denoises `.wav` files (no Python needed).
2. `deepFilter` / `df/enhance.py` — Python framework (PyTorch) for enhancement/eval/training.

Standard commands live in `README.md`, the per-crate `Cargo.toml`s, `DeepFilterNet/pyproject.toml`
(poetry `poe` tasks), and the CI workflows under `.github/workflows/` (`test_df.yml`,
`rust_lint.yml`, `python_lint.yml`). Prefer those as the source of truth.

### Environment specifics (already applied by the startup update script)

- **Python must be 3.11**, not the system default 3.12. The pinned `numpy==1.24.4` in
  `poetry.lock` has no cp312 wheel and fails to build on 3.12. The poetry venv is created
  with `python3.11` (installed via the deadsnakes PPA) and lives at `DeepFilterNet/.venv`
  (in-project).
- **`numpy` must stay `<2.0`.** `poe install-dnsmos-utils` (librosa/numba) pulls numpy 2.x,
  which breaks torch 2.1 (`Failed to initialize NumPy: _ARRAY_API not found`). The update
  script re-pins `numpy==1.26.4` as its last step; keep any manual reinstall of dnsmos/librosa
  deps followed by re-pinning numpy `<2.0`.
- Poetry is installed at `~/.local/bin`; ensure it is on `PATH` (it is added in `~/.bashrc`).
- The Python package is installed with `--no-root`, so set `PYTHONPATH=$PWD/DeepFilterNet`
  (repo root: `PYTHONPATH=/workspace/DeepFilterNet`) when running `df/*.py` scripts directly.
  `poetry run deepFilter ...` also works (poetry 2.x runs the uninstalled console script,
  printing a harmless deprecation warning).

### Running the product (from repo root)

- Rust CLI (build once, then run):
  `cargo run -p deep_filter --profile=release --features=tract,bin,wav-utils,transforms --bin deep-filter -- ./assets/noisy_snr0.wav -m ./models/DeepFilterNet3_onnx.tar.gz -o out_DeepFilterNet3`
- Python enhancement:
  `cd DeepFilterNet && PYTHONPATH=$PWD poetry run deepFilter --no-delay-compensation ../assets/noisy_snr0.wav -m DeepFilterNet3 -o ../out_py`
  (pretrained models auto-download to `~/.cache/DeepFilterNet`).

### Tests / lint

- Python model tests: `cd DeepFilterNet && poetry run python df/scripts/test_df.py` (needs the `eval` extra: pesq/pystoi/scipy).
- DNSMOS eval sample: `cd DeepFilterNet && poetry run python df/scripts/dnsmos_dns5.py eval-sample ../assets/noisy_snr0.wav -t <targets>` (needs dnsmos utils + numpy `<2.0`).
- Rust tests: `cargo test --all-features -p deep_filter` (needs `libhdf5-dev`, already installed).
- Python lint (CI installs these standalone, not via poetry): `flake8`, `black --check --diff .`, `isort . --check --diff`. Note: with current tool versions there are pre-existing findings (flake8 F824/E226, one black file) that are not from environment setup.

### Optional components (not set up by default)

- `pyDF-data` / training (`-E train`) and Rust `cargo build -p DeepFilterDataLoader` need `libhdf5-dev` (installed).
- The `df-demo` GUI and `cargo fmt`/`clippy` require the **nightly** Rust toolchain
  (`rustup toolchain install nightly`) plus `libasound2-dev`; the GUI also needs a display.
