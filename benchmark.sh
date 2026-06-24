#!/usr/bin/env bash
# benchmark.sh — Run the full PoC benchmark suite on GPU
set -euo pipefail

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "Adaptive Attention Token Reduction — Benchmark"
echo "=============================================="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Host: $(hostname)"
echo ""

# Check GPU availability
if command -v nvidia-smi &>/dev/null; then
    echo "--- GPU Info ---"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "WARNING: nvidia-smi not found. Running CPU-only mode."
fi

# Check Python and torch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
echo ""

# --- Run 1: Default decay_factor=0.95 ---
echo "=== Run 1: decay_factor=0.95 (default) ==="
python run_poc.py \
    --decay-factor 0.95 \
    2>&1 | tee "$RESULTS_DIR/run_decay095.log"

cp "$RESULTS_DIR/poc_results.csv" "$RESULTS_DIR/results_decay095.csv"
echo ""

# --- Run 2: decay_factor=1.0 (no recency bias) ---
echo "=== Run 2: decay_factor=1.0 (no recency bias) ==="
python run_poc.py \
    --decay-factor 1.0 \
    2>&1 | tee "$RESULTS_DIR/run_decay100.log"

cp "$RESULTS_DIR/poc_results.csv" "$RESULTS_DIR/results_decay100.csv"
echo ""

# --- Run 3: decay_factor=0.90 (aggressive recency) ---
echo "=== Run 3: decay_factor=0.90 (aggressive recency) ==="
python run_poc.py \
    --decay-factor 0.90 \
    2>&1 | tee "$RESULTS_DIR/run_decay090.log"

cp "$RESULTS_DIR/poc_results.csv" "$RESULTS_DIR/results_decay090.csv"
echo ""

# --- Run 4: Tighter token budget (60%) ---
echo "=== Run 4: token_budget_ratio=0.6 (tighter budget) ==="
python run_poc.py \
    --decay-factor 0.95 \
    --token-budget-ratio 0.6 \
    2>&1 | tee "$RESULTS_DIR/run_budget060.log"

cp "$RESULTS_DIR/poc_results.csv" "$RESULTS_DIR/results_budget060.csv"
echo ""

# --- Summary ---
echo "=============================================="
echo "Benchmark complete!"
echo "Results saved in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"
echo ""
echo "Key files:"
echo "  results_decay095.csv  — Default run (decay=0.95)"
echo "  results_decay100.csv  — No recency bias (decay=1.0)"
echo "  results_decay090.csv  — Aggressive recency (decay=0.90)"
echo "  results_budget060.csv — Tighter budget (ratio=0.6)"
echo "=============================================="
