#!/bin/bash
set -e
echo "=== VM Setup Starting ==="
echo "Hostname: $(hostname)"
uname -a
echo "---"
lspci | grep -i nvidia || echo "No NVIDIA GPU detected in lspci yet"
echo "---"
# Install NVIDIA drivers
apt-get update -y
apt-get install -y linux-headers-$(uname -r) ubuntu-drivers-common
ubuntu-drivers install --gpgpu
# Install Python 3.11
apt-get install -y python3.11 python3.11-venv python3-pip python3.11-dev git
ln -sf /usr/bin/python3.11 /usr/bin/python
# Install Docker
curl -fsSL https://get.docker.com | sh
usermod -aG docker benchuser
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
echo "=== Setup Complete ==="
