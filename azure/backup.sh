#!/usr/bin/env bash
# backup.sh — Archive images, results, and artifacts to Azure Blob Storage
# Enables fast restart without rebuilding models or re-running experiments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Configuration (override via environment) ---
STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-attnmatterarchive}"
STORAGE_RG="${AZURE_STORAGE_RG:-attentionmatter-archive}"
LOCATION="${AZURE_LOCATION:-eastus}"
CONTAINER_NAME="${AZURE_CONTAINER_NAME:-poc-artifacts}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE_PREFIX="archive/${TIMESTAMP}"

echo "=============================================="
echo "Adaptive Attention PoC — Backup to Azure Storage"
echo "=============================================="
echo "Storage Account: $STORAGE_ACCOUNT"
echo "Container:       $CONTAINER_NAME"
echo "Archive prefix:  $ARCHIVE_PREFIX"
echo "Timestamp:       $TIMESTAMP"
echo ""

# --- Ensure storage account exists ---
echo "[1/5] Ensuring resource group and storage account exist..."
az group create --name "$STORAGE_RG" --location "$LOCATION" --output none 2>/dev/null || true

ACCOUNT_EXISTS=$(az storage account check-name --name "$STORAGE_ACCOUNT" --query "nameAvailable" -o tsv)
if [[ "$ACCOUNT_EXISTS" == "true" ]]; then
    echo "  Creating storage account '$STORAGE_ACCOUNT'..."
    az storage account create \
        --name "$STORAGE_ACCOUNT" \
        --resource-group "$STORAGE_RG" \
        --location "$LOCATION" \
        --sku Standard_LRS \
        --kind StorageV2 \
        --output none
fi

# Get connection string
CONN_STR=$(az storage account show-connection-string \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$STORAGE_RG" \
    --query connectionString -o tsv)

# --- Ensure container exists ---
echo "[2/5] Ensuring blob container '$CONTAINER_NAME' exists..."
az storage container create \
    --name "$CONTAINER_NAME" \
    --connection-string "$CONN_STR" \
    --output none 2>/dev/null || true

# --- Upload results ---
echo "[3/5] Uploading results and logs..."
if [[ -d "$PROJECT_DIR/results" ]] && ls "$PROJECT_DIR/results"/*.{csv,log,png} 2>/dev/null; then
    az storage blob upload-batch \
        --destination "$CONTAINER_NAME" \
        --source "$PROJECT_DIR/results" \
        --destination-path "$ARCHIVE_PREFIX/results" \
        --connection-string "$CONN_STR" \
        --overwrite \
        --output none
    echo "  Uploaded results/ → $ARCHIVE_PREFIX/results/"
else
    echo "  No results found. Skipping."
fi

# --- Upload Docker image (if available) ---
echo "[4/5] Archiving Docker image (if built)..."
IMAGE_NAME="attn-benchmark"
IMAGE_TAR="$PROJECT_DIR/.cache/attn-benchmark-${TIMESTAMP}.tar.gz"

if docker image inspect "$IMAGE_NAME" &>/dev/null; then
    mkdir -p "$PROJECT_DIR/.cache"
    echo "  Exporting Docker image to tar..."
    docker save "$IMAGE_NAME" | gzip > "$IMAGE_TAR"
    echo "  Uploading image archive..."
    az storage blob upload \
        --container-name "$CONTAINER_NAME" \
        --file "$IMAGE_TAR" \
        --name "$ARCHIVE_PREFIX/docker/${IMAGE_NAME}.tar.gz" \
        --connection-string "$CONN_STR" \
        --output none
    rm -f "$IMAGE_TAR"
    echo "  Uploaded Docker image → $ARCHIVE_PREFIX/docker/"
else
    echo "  Docker image '$IMAGE_NAME' not found locally. Skipping."
fi

# --- Upload model cache (embeddings + tokenizer) ---
echo "[5/5] Uploading model cache for fast restart..."
MODEL_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODELS_TO_ARCHIVE=(
    "sentence-transformers/all-MiniLM-L6-v2"
)

for MODEL_PATH in "${MODELS_TO_ARCHIVE[@]}"; do
    SAFE_NAME=$(echo "$MODEL_PATH" | tr '/' '_')
    MODEL_DIR="$MODEL_CACHE_DIR/hub/models--${SAFE_NAME}"
    if [[ -d "$MODEL_DIR" ]]; then
        echo "  Archiving model: $MODEL_PATH..."
        TAR_FILE="$PROJECT_DIR/.cache/${SAFE_NAME}.tar.gz"
        tar -czf "$TAR_FILE" -C "$MODEL_CACHE_DIR/hub" "models--${SAFE_NAME}"
        az storage blob upload \
            --container-name "$CONTAINER_NAME" \
            --file "$TAR_FILE" \
            --name "$ARCHIVE_PREFIX/models/${SAFE_NAME}.tar.gz" \
            --connection-string "$CONN_STR" \
            --output none
        rm -f "$TAR_FILE"
        echo "  Uploaded → $ARCHIVE_PREFIX/models/${SAFE_NAME}.tar.gz"
    else
        echo "  Model cache for '$MODEL_PATH' not found. Skipping."
    fi
done

# --- Upload source snapshot ---
echo ""
echo "Uploading source code snapshot..."
SRC_TAR="$PROJECT_DIR/.cache/source-${TIMESTAMP}.tar.gz"
mkdir -p "$PROJECT_DIR/.cache"
tar -czf "$SRC_TAR" \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='.cache' \
    --exclude='results' \
    -C "$PROJECT_DIR" .
az storage blob upload \
    --container-name "$CONTAINER_NAME" \
    --file "$SRC_TAR" \
    --name "$ARCHIVE_PREFIX/source/attentionmatter-${TIMESTAMP}.tar.gz" \
    --connection-string "$CONN_STR" \
    --output none
rm -f "$SRC_TAR"
echo "  Uploaded source snapshot."

# --- Write manifest ---
echo ""
echo "Writing archive manifest..."
MANIFEST=$(cat <<EOF
{
    "timestamp": "$TIMESTAMP",
    "archive_prefix": "$ARCHIVE_PREFIX",
    "storage_account": "$STORAGE_ACCOUNT",
    "container": "$CONTAINER_NAME",
    "contents": {
        "results": "$ARCHIVE_PREFIX/results/",
        "docker_image": "$ARCHIVE_PREFIX/docker/",
        "models": "$ARCHIVE_PREFIX/models/",
        "source": "$ARCHIVE_PREFIX/source/"
    },
    "config": {
        "decay_factor": 0.95,
        "token_budget_ratio": 0.8,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "mistralai/Mistral-7B-Instruct-v0.2"
    }
}
EOF
)
echo "$MANIFEST" > "/tmp/manifest-${TIMESTAMP}.json"
az storage blob upload \
    --container-name "$CONTAINER_NAME" \
    --file "/tmp/manifest-${TIMESTAMP}.json" \
    --name "$ARCHIVE_PREFIX/manifest.json" \
    --connection-string "$CONN_STR" \
    --output none
rm -f "/tmp/manifest-${TIMESTAMP}.json"

# Also upload as "latest" pointer
echo "$ARCHIVE_PREFIX" > "/tmp/latest.txt"
az storage blob upload \
    --container-name "$CONTAINER_NAME" \
    --file "/tmp/latest.txt" \
    --name "latest.txt" \
    --connection-string "$CONN_STR" \
    --overwrite \
    --output none
rm -f "/tmp/latest.txt"

echo ""
echo "=============================================="
echo "Backup complete!"
echo ""
echo "Archive:  $ARCHIVE_PREFIX"
echo "Account:  $STORAGE_ACCOUNT"
echo "Container: $CONTAINER_NAME"
echo ""
echo "To restore: ./azure/restore.sh"
echo "=============================================="
