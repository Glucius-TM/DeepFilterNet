#!/usr/bin/env python3
"""Micro-benchmark for the libdf (pyDF) Rust<->Python boundary.

This measures the per-call latency of the STFT/ISTFT and feature transforms that
`df.enhance` drives through the native `libdf` module. It is meant to establish a
baseline before/after optimisation work (e.g. reducing memory copies across the
Rust<->Python boundary) and to give a rough real-time factor (RTF) for the DSP
part alone (i.e. excluding the neural network / tract inference).

Usage:
    python scripts/bench_libdf.py [--seconds 5] [--sr 48000] [--iters 200]

It only needs `numpy` and the `libdf` wheel installed (no torch required).
"""

import argparse
import statistics
import time
from typing import Callable, List

import numpy as np

import libdf


def _time_call(fn: Callable[[], object], iters: int, warmup: int = 5) -> List[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def _report(name: str, samples: List[float], audio_seconds: float) -> None:
    mean = statistics.mean(samples)
    p50 = statistics.median(samples)
    p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]
    rtf = mean / audio_seconds if audio_seconds > 0 else float("nan")
    print(
        f"{name:<16} mean={mean * 1e3:8.3f} ms  p50={p50 * 1e3:8.3f} ms  "
        f"p95={p95 * 1e3:8.3f} ms  rtf={rtf:.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr", type=int, default=48000, help="Sampling rate")
    parser.add_argument("--fft-size", type=int, default=960)
    parser.add_argument("--hop-size", type=int, default=480)
    parser.add_argument("--nb-erb", type=int, default=32)
    parser.add_argument("--nb-df", type=int, default=96)
    parser.add_argument("--seconds", type=float, default=5.0, help="Audio length per call")
    parser.add_argument("--iters", type=int, default=200, help="Iterations to time")
    parser.add_argument("--alpha", type=float, default=0.9)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    df = libdf.DF(args.sr, args.fft_size, args.hop_size, args.nb_erb)
    erb_fb = df.erb_widths()

    n_samples = int(args.seconds * args.sr)
    # Round to a whole number of hops.
    n_samples -= n_samples % args.hop_size
    audio = rng.standard_normal((1, n_samples)).astype(np.float32)
    audio_seconds = n_samples / args.sr

    spec = df.analysis(audio)  # [C, T, F]
    spec_df = np.ascontiguousarray(spec[..., : args.nb_df])

    print(
        f"libdf boundary benchmark | numpy {np.__version__} | "
        f"sr={args.sr} fft={args.fft_size} hop={args.hop_size} "
        f"audio={audio_seconds:.2f}s frames={spec.shape[1]} iters={args.iters}"
    )
    print("-" * 78)

    _report("analysis", _time_call(lambda: df.analysis(audio), args.iters), audio_seconds)
    _report("synthesis", _time_call(lambda: df.synthesis(spec), args.iters), audio_seconds)
    _report("erb", _time_call(lambda: libdf.erb(spec, erb_fb), args.iters), audio_seconds)
    _report(
        "erb_norm",
        _time_call(lambda: libdf.erb_norm(libdf.erb(spec, erb_fb), args.alpha), args.iters),
        audio_seconds,
    )
    _report(
        "unit_norm",
        _time_call(lambda: libdf.unit_norm(spec_df, args.alpha), args.iters),
        audio_seconds,
    )

    # Full feature pipeline as used by df.enhance (DSP only, no neural network).
    def full_features():
        s = df.analysis(audio)
        libdf.erb_norm(libdf.erb(s, erb_fb), args.alpha)
        libdf.unit_norm(np.ascontiguousarray(s[..., : args.nb_df]), args.alpha)

    _report("features(all)", _time_call(full_features, args.iters), audio_seconds)

    # Single-hop latency, relevant for low-latency streaming: process one frame
    # (analysis + synthesis) at a time. `rtf` here is per-hop time / hop duration.
    hop_audio = rng.standard_normal((1, args.hop_size)).astype(np.float32)
    hop_seconds = args.hop_size / args.sr

    def stream_hop():
        df.synthesis(df.analysis(hop_audio))

    print("-" * 78)
    _report("stream(1 hop)", _time_call(stream_hop, max(args.iters, 1000)), hop_seconds)


if __name__ == "__main__":
    main()
