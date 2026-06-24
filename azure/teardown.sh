#!/usr/bin/env bash
# teardown.sh — Delete Azure resources after benchmarking
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load connection info
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: azure/.env not found."
    exit 1
fi

echo "=== Azure Resource Teardown ==="
echo "Resource Group: $AZURE_RG"
echo "VM:             $AZURE_VM_NAME"
echo ""
echo "WARNING: This will DELETE all resources in resource group '$AZURE_RG'."
echo ""
read -p "Are you sure? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 0
fi

echo "Deleting resource group '$AZURE_RG'..."
az group delete \
    --name "$AZURE_RG" \
    --yes \
    --no-wait

echo ""
echo "Resource group deletion initiated (async)."
echo "Check status: az group show --name $AZURE_RG --query properties.provisioningState"
echo ""

# Clean up local env file
rm -f "$SCRIPT_DIR/.env"
echo "Removed azure/.env"
echo "Done."
