#!/usr/bin/env bash
# Bootstrap the GPU arm on a rented NVIDIA box (e.g. RTX 5090 on Runpod/CloudRift).
# The CPU pipeline runs anywhere; only `--features gpu` (the M5 kernel) needs CUDA.
set -euo pipefail

echo "== checking CUDA toolchain =="
if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not found — this is not a GPU host." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if ! command -v nvcc >/dev/null; then
  echo "WARN: nvcc not on PATH. cudarc needs the CUDA toolkit to build." >&2
  echo "      e.g. export PATH=/usr/local/cuda/bin:\$PATH" >&2
fi

echo "== building (release, gpu feature) =="
cargo build --release --features gpu

echo "== end-to-end smoke test =="
fifo=./target/release/fifo
OUT=${OUT:-data}
$fifo gen   --clients 20000 --days 400 --whales 20 --out "$OUT/tradebook"
$fifo pack  --tradebook "$OUT/tradebook" --out "$OUT/compute.fifopack"
$fifo checkpoint --tradebook "$OUT/tradebook" --packed "$OUT/compute.fifopack" --out "$OUT/checkpoints"
$fifo bench --tradebook "$OUT/tradebook" --packed "$OUT/compute.fifopack" --checkpoints "$OUT/checkpoints"

echo
echo "Done. The bench output above includes the GPU arm (disk/H2D/kernel/D2H"
echo "breakout) and validation of the GPU result against the CPU oracle."
