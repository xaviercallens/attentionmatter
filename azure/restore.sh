#!/usr/bin/env bash
# restore.sh — Restore archived artifacts from Azure Blob Storage for fast restart
# Downloads model caches, Docker images, and previous results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration ---
STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-attnmatterarchive}"
STORAGE_RG="${AZURE_STORAGE_RG:-attentionmatter-archive}"
CONTAINER_NAME="${AZURE_CONTAINER_NAME:-poc-artifacts}"
RESTORE_DIR="$PROJECT_DIR/.cache/restore"
TARGET="${1:-latest}"  # pass a specific timestamp or "latest"

echo "=============================================="
echo "Adaptive Attention PoC — Restore from Azure Storage"
echo "=============================================="
echo "Storage Account: $STORAGE_ACCOUNT"
echo "Target:          $TARGET"
echo ""

# Get connection string
CONN_STR=$(az storage account show-connection-string \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$STORAGE_RG" \
    --query connectionString -o tsv)

# Determine archive prefix
if [[ "$TARGET" == "latest" ]]; then
    echo "[1/5] Finding latest archive..."
    az storage blob download \
        --container-name "$CONTAINER_NAME" \
        --name "latest.txt" \
        --file "/tmp/latest_prefix.txt" \
        --connection-string "$CONN_STR" \
        --output none
    ARCHIVE_PREFIX=$(cat /tmp/latest_prefix.txt | tr -d '\n')
    rm -f /tmp/latest_prefix.txt
else
    ARCHIVE_PREFIX="archive/$TARGET"
fi

echo "  Archive prefix: $ARCHIVE_PREFIX"
mkdir -p "$RESTORE_DIR"

# --- Download manifest ---
echo "[2/5] Downloading manifest..."
az storage blob download \
    --container-name "$CONTAINER_NAME" \
    --name "$ARCHIVE_PREFIX/manifest.json" \
    --file "$RESTORE_DIR/manifest.json" \
    --connection-string "$CONN_STR" \
    --output none
cat "$RESTORE_DIR/manifest.json"
echo ""

# --- Restore results ---
echo "[3/5] Restoring results..."
mkdir -p "$PROJECT_DIR/results"
az storage blob download-batch \
    --destination "$PROJECT_DIR/results" \
    --source "$CONTAINER_NAME" \
    --pattern "$ARCHIVE_PREFIX/results/*" \
    --connection-string "$CONN_STR" \
    --output none 2>/dev/null || echo "  No results found in archive."
# Flatten: move files from nested archive path to results/
find "$PROJECT_DIR/results" -name "*.csv" -path "*/archive/*" -exec mv {} "$PROJECT_DIR/results/" \; 2>/dev/null || true
find "$PROJECT_DIR/results" -name "*.log" -path "*/archive/*" -exec mv {} "$PROJECT_DIR/results/" \; 2>/dev/null || true
find "$PROJECT_DIR/results" -name "*.png" -path "*/archive/*" -exec mv {} "$PROJECT_DIR/results/" \; 2>/dev/null || true
find "$PROJECT_DIR/results" -type d -empty -delete 2>/dev/null || true
echo "  Results restored to: results/"

# --- Restore Docker image ---
echo "[4/5] Restoring Docker image..."
DOCKER_BLOB="$ARCHIVE_PREFIX/docker/attn-benchmark.tar.gz"
DOCKER_TAR="$RESTORE_DIR/attn-benchmark.tar.gz"

BLOB_EXISTS=$(az storage blob exists \
    --container-name "$CONTAINER_NAME" \
    --name "$DOCKER_BLOB" \
    --connection-string "$CONN_STR" \
    --query "exists" -o tsv 2>/dev/null || echo "false")

if [[ "$BLOB_EXISTS" == "true" ]]; then
    az storage blob download \
        --container-name "$CONTAINER_NAME" \
        --name "$DOCKER_BLOB" \
        --file "$DOCKER_TAR" \
        --connection-string "$CONN_STR" \
        --output none
    echo "  Loading Docker image..."
    docker load < "$DOCKER_TAR"
    rm -f "$DOCKER_TAR"
    echo "  Docker image 'attn-benchmark' restored."
else
    echo "  No Docker image in archive. Skipping."
fi

# --- Restore model caches ---
echo "[5/5] Restoring model caches..."
MODEL_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub"
mkdir -p "$MODEL_CACHE_DIR"

# List model blobs
MODEL_BLOBS=$(az storage blob list \
    --container-name "$CONTAINER_NAME" \
    --prefix "$ARCHIVE_PREFIX/models/" \
    --connection-string "$CONN_STR" \
    --query "[].name" -o tsv 2>/dev/null || echo "")

for BLOB in $MODEL_BLOBS; do
    FILENAME=$(basename "$BLOB")
    echo "  Downloading $FILENAME..."
    az storage blob download \
        --container-name "$CONTAINER_NAME" \
        --name "$BLOB" \
        --file "$RESTORE_DIR/$FILENAME" \
        --connection-string "$CONN_STR" \
        --output none
    echo "  Extracting to model cache..."
    tar -xzf "$RESTORE_DIR/$FILENAME" -C "$MODEL_CACHE_DIR"
    rm -f "$RESTORE_DIR/$FILENAME"
done

# --- Cleanup ---
rm -rf "$RESTORE_DIR"

echo ""
echo "=============================================="
echo "Restore complete!"
echo ""
echo "Restored from: $ARCHIVE_PREFIX"
echo ""
echo "You can now run:"
echo "  python run_poc.py --dummy-llm --real-embeddings  # CPU (models cached)"
echo "  ./azure/deploy.sh docker                         # Azure (image restored)"
echo "  python run_poc.py                                # GPU (if local GPU)"
echo "=============================================="
