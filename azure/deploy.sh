#!/usr/bin/env bash
# deploy.sh — Upload code to Azure VM and run the benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load connection info
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: azure/.env not found. Run ./azure/provision.sh first."
    exit 1
fi

VM_IP="${AZURE_VM_IP}"
VM_USER="${AZURE_ADMIN_USER}"
REMOTE_DIR="/home/$VM_USER/attentionmatter"

echo "=== Deploying to Azure VM ==="
echo "Host: $VM_USER@$VM_IP"
echo "Remote dir: $REMOTE_DIR"
echo ""

# --- Upload source code ---
echo "[1/4] Uploading project files..."
ssh "$VM_USER@$VM_IP" "mkdir -p $REMOTE_DIR"

rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '*.pyc' \
    --exclude 'results/' \
    --exclude '.kiro/' \
    --exclude 'azure/.env' \
    "$PROJECT_DIR/" "$VM_USER@$VM_IP:$REMOTE_DIR/"

echo ""

# --- Choose execution mode ---
MODE="${1:-docker}"

if [[ "$MODE" == "docker" ]]; then
    echo "[2/4] Building Docker image on VM..."
    ssh "$VM_USER@$VM_IP" "cd $REMOTE_DIR && docker build -t attn-benchmark ."

    echo "[3/4] Running benchmark in Docker (GPU enabled)..."
    ssh "$VM_USER@$VM_IP" "cd $REMOTE_DIR && docker run --gpus all \
        -v $REMOTE_DIR/results:/workspace/results \
        attn-benchmark"

elif [[ "$MODE" == "bare" ]]; then
    echo "[2/4] Installing dependencies on VM..."
    ssh "$VM_USER@$VM_IP" "cd $REMOTE_DIR && \
        python -m pip install --upgrade pip && \
        pip install -r requirements-gpu.txt"

    echo "[3/4] Running benchmark directly..."
    ssh "$VM_USER@$VM_IP" "cd $REMOTE_DIR && bash benchmark.sh"

else
    echo "ERROR: Unknown mode '$MODE'. Use 'docker' or 'bare'."
    exit 1
fi

# --- Download results ---
echo "[4/4] Downloading results..."
mkdir -p "$PROJECT_DIR/results"
rsync -avz "$VM_USER@$VM_IP:$REMOTE_DIR/results/" "$PROJECT_DIR/results/"

echo ""
echo "=============================================="
echo "Benchmark complete! Results downloaded to: results/"
echo ""
ls -la "$PROJECT_DIR/results/"
echo "=============================================="
