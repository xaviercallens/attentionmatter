#!/usr/bin/env bash
# provision.sh — Provision an Azure GPU VM for the PoC benchmark
# Prerequisites: Azure CLI (az) authenticated, sufficient quota for GPU VMs
set -euo pipefail

# --- Configuration (override via environment) ---
RESOURCE_GROUP="${AZURE_RG:-attentionmatter-benchmark}"
LOCATION="${AZURE_LOCATION:-eastus}"
VM_NAME="${AZURE_VM_NAME:-attn-bench-vm}"
VM_SIZE="${AZURE_VM_SIZE:-Standard_NC6s_v3}"  # 1x V100 16GB (~$3/hr)
IMAGE="${AZURE_IMAGE:-Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest}"
ADMIN_USER="${AZURE_ADMIN_USER:-benchuser}"
SSH_KEY_PATH="${AZURE_SSH_KEY:-~/.ssh/id_rsa.pub}"
DISK_SIZE="${AZURE_DISK_SIZE:-128}"  # GB

echo "=== Azure GPU VM Provisioning ==="
echo "Resource Group: $RESOURCE_GROUP"
echo "Location:       $LOCATION"
echo "VM Name:        $VM_NAME"
echo "VM Size:        $VM_SIZE"
echo "Image:          $IMAGE"
echo ""

# --- Create Resource Group ---
echo "[1/5] Creating resource group..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

# --- Create VM ---
echo "[2/5] Creating GPU VM (this may take 2-5 minutes)..."
az vm create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --size "$VM_SIZE" \
    --image "$IMAGE" \
    --admin-username "$ADMIN_USER" \
    --ssh-key-values "$SSH_KEY_PATH" \
    --os-disk-size-gb "$DISK_SIZE" \
    --public-ip-sku Standard \
    --output table

# --- Get public IP ---
echo "[3/5] Retrieving public IP..."
PUBLIC_IP=$(az vm show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --show-details \
    --query publicIps \
    --output tsv)
echo "VM Public IP: $PUBLIC_IP"

# --- Open SSH port (if not already open) ---
echo "[4/5] Ensuring SSH port is open..."
az vm open-port \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --port 22 \
    --priority 1001 \
    --output none 2>/dev/null || true

# --- Install NVIDIA drivers + Docker via cloud-init ---
echo "[5/5] Installing NVIDIA drivers and Docker on VM..."
az vm run-command invoke \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --command-id RunShellScript \
    --scripts '
        set -e
        # Update system
        sudo apt-get update -y
        sudo apt-get upgrade -y

        # Install NVIDIA drivers
        sudo apt-get install -y linux-headers-$(uname -r)
        sudo apt-get install -y nvidia-driver-535 nvidia-utils-535

        # Install Docker
        curl -fsSL https://get.docker.com | sudo sh
        sudo usermod -aG docker '"$ADMIN_USER"'

        # Install NVIDIA Container Toolkit
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update -y
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker

        # Install Python 3.11 (for non-Docker usage)
        sudo apt-get install -y python3.11 python3.11-venv python3-pip python3.11-dev
        sudo ln -sf /usr/bin/python3.11 /usr/bin/python

        echo "Setup complete. Reboot recommended for NVIDIA drivers."
    ' \
    --output table

echo ""
echo "=============================================="
echo "Provisioning complete!"
echo ""
echo "VM:       $VM_NAME"
echo "IP:       $PUBLIC_IP"
echo "User:     $ADMIN_USER"
echo ""
echo "SSH:      ssh $ADMIN_USER@$PUBLIC_IP"
echo ""
echo "IMPORTANT: The VM may need a reboot for NVIDIA drivers:"
echo "  az vm restart --resource-group $RESOURCE_GROUP --name $VM_NAME"
echo ""
echo "Next steps:"
echo "  1. Reboot VM:  az vm restart -g $RESOURCE_GROUP -n $VM_NAME"
echo "  2. Deploy:     ./azure/deploy.sh"
echo "  3. Teardown:   ./azure/teardown.sh"
echo "=============================================="

# Save connection info for other scripts
cat > azure/.env <<EOF
AZURE_RG=$RESOURCE_GROUP
AZURE_VM_NAME=$VM_NAME
AZURE_VM_IP=$PUBLIC_IP
AZURE_ADMIN_USER=$ADMIN_USER
AZURE_LOCATION=$LOCATION
EOF

echo "Connection info saved to azure/.env"
