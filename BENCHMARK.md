# Benchmark Guide — Adaptive Attention Token Reduction PoC

## Overview

This guide covers running the PoC benchmark on Azure GPU infrastructure. The
benchmark runs 7 scenarios across 4 context-management strategies (No-Pruning,
Sliding-Window, A3TK-Heuristic, Adaptive) under multiple parameter configurations
to measure token reduction vs answer quality.

## Prerequisites

- Azure CLI (`az`) installed and authenticated (`az login`)
- SSH key pair (default: `~/.ssh/id_rsa.pub`)
- GPU quota in your Azure subscription for `Standard_NC6s_v3` (or alternative)
- rsync and ssh available locally

### Quick Check

```bash
az account show  # verify login
az vm list-skus --location eastus --size Standard_NC6 --output table  # verify quota
```

## Cost Estimate

| VM Size | GPU | Approx Cost | Benchmark Duration |
|---------|-----|-------------|-------------------|
| Standard_NC6s_v3 | 1x V100 16GB | ~$3.06/hr | ~30-45 min |
| Standard_NC4as_T4_v3 | 1x T4 16GB | ~$0.53/hr | ~60-90 min |
| Standard_NC24ads_A100_v4 | 1x A100 80GB | ~$3.67/hr | ~15-20 min |

The default is `Standard_NC6s_v3` (V100). For lower cost, override with:
```bash
export AZURE_VM_SIZE=Standard_NC4as_T4_v3
```

## Quick Start (Full Workflow)

```bash
# 1. Provision Azure GPU VM (~5 minutes)
./azure/provision.sh

# 2. Reboot VM for NVIDIA drivers to load
az vm restart -g attentionmatter-benchmark -n attn-bench-vm
sleep 60

# 3. Deploy code and run benchmark (~30-45 minutes)
./azure/deploy.sh docker

# 4. Review results locally
cat results/results_decay095.csv
cat results/run_decay095.log

# 5. Tear down Azure resources (stops billing!)
./azure/teardown.sh
```

## Detailed Steps

### Step 1: Provision

```bash
# Default configuration
./azure/provision.sh

# Or override parameters
AZURE_LOCATION=westus2 AZURE_VM_SIZE=Standard_NC4as_T4_v3 ./azure/provision.sh
```

The script:
- Creates a resource group `attentionmatter-benchmark`
- Provisions a GPU VM with Ubuntu 22.04
- Installs NVIDIA drivers (535), Docker, and nvidia-container-toolkit
- Saves connection info to `azure/.env`

### Step 2: Reboot & Verify

After provisioning, reboot so NVIDIA drivers initialize:

```bash
az vm restart -g attentionmatter-benchmark -n attn-bench-vm
sleep 60

# Verify GPU is accessible
source azure/.env
ssh $AZURE_ADMIN_USER@$AZURE_VM_IP "nvidia-smi"
```

### Step 3: Deploy & Run

Two execution modes:

```bash
# Docker mode (recommended — isolated, reproducible)
./azure/deploy.sh docker

# Bare-metal mode (faster if deps already installed)
./azure/deploy.sh bare
```

The deploy script:
1. Uploads project files via rsync
2. Builds Docker image (or installs pip deps)
3. Runs `benchmark.sh` which executes 4 parameter configurations
4. Downloads results to local `results/` directory

### Step 4: Review Results

```bash
# Summary table
cat results/run_decay095.log | grep -A 20 "RESULTS"

# CSV for further analysis
cat results/results_decay095.csv

# Compare decay factor impact
diff results/results_decay095.csv results/results_decay100.csv
```

### Step 5: Teardown

```bash
./azure/teardown.sh
```

This deletes the entire resource group and all associated resources.

## Benchmark Configurations

The `benchmark.sh` script runs four configurations:

| Run | Decay Factor | Budget Ratio | Purpose |
|-----|-------------|--------------|---------|
| 1   | 0.95        | 0.80         | Default — balanced recency + relevance |
| 2   | 1.00        | 0.80         | No recency bias — pure semantic similarity |
| 3   | 0.90        | 0.80         | Aggressive recency — strong older-item penalty |
| 4   | 0.95        | 0.60         | Tighter budget — forces more aggressive pruning |

## Running Locally (Without Azure)

### With real embeddings, dummy LLM (no GPU needed)

```bash
pip install -r requirements.txt
python run_poc.py --dummy-llm
```

This uses real sentence-transformer embeddings for semantic scoring but a
deterministic stub for answer generation. Useful for verifying token reduction
behavior without GPU hardware.

### Dry run (verify config only)

```bash
python run_poc.py --dry-run
```

### With real LLM (requires GPU with 16GB+ VRAM)

```bash
pip install -r requirements-gpu.txt
python run_poc.py
```

## Expected Results

With real embeddings and LLM, we expect:

| Strategy | Avg Token Reduction | Pass Rate (key fact) |
|----------|--------------------|--------------------|
| No-Pruning | 0% (baseline) | ~100% |
| Sliding-Window | ~80% | ~15-30% |
| A3TK-Heuristic | ~20-30% | ~70-85% |
| Adaptive (ours) | ~30-50% | ~85-100% |

The Adaptive strategy should demonstrate the best quality-to-token-reduction
trade-off: significant savings while preserving critical information.

### Verified Local Results (real embeddings, tight budget)

With `token_budget_ratio=0.08` (forcing pruning) and real sentence-transformer
embeddings:

| Scenario | No-Pruning tokens | Adaptive tokens | Reduction | Key Fact Preserved |
|----------|-------------------|-----------------|-----------|-------------------|
| flight_booking_memory | 1001 | 655 | 34.6% | Yes (XYZ789) |
| irrelevant_heavy | 1486 | 654 | 56.0% | Yes (ACC-9182736) |
| preference_recall | 793 | 653 | 17.7% | Yes (vegetarian) |
| cross_session_name | 234 | 234 | 0.0% | Yes (Alexander) |
| multi_fact | 610 | 610 | 0.0% | Yes (555-0142) |

Key findings:
- On scenarios with many irrelevant turns, 34-56% token reduction achieved.
- All key facts preserved in every scenario (100% pass rate).
- Smaller scenarios (already within budget) show 0% reduction — expected behavior.

## Troubleshooting

### GPU not detected on VM

```bash
# Check driver installation
ssh user@vm "dmesg | grep -i nvidia"
# Reinstall if needed
ssh user@vm "sudo apt-get install -y nvidia-driver-535 && sudo reboot"
```

### Out of GPU memory

Edit `azure/benchmark-config.json` and set `"use_4bit": true` (already default).
Or switch to a larger VM size.

### Model download fails

The Dockerfile pre-downloads the embedding model. For the LLM, it downloads on
first run. If HuggingFace is slow, set a mirror:
```bash
ssh user@vm "export HF_ENDPOINT=https://hf-mirror.com && cd attentionmatter && bash benchmark.sh"
```

### Quota errors during provisioning

```
The subscription does not have enough quota for the requested VM size
```

Try a different region or VM size:
```bash
AZURE_LOCATION=westeurope AZURE_VM_SIZE=Standard_NC4as_T4_v3 ./azure/provision.sh
```

Or request a quota increase in the Azure portal under
Subscription → Usage + quotas.

## File Reference

```
azure/
├── provision.sh          # Create GPU VM and install drivers
├── deploy.sh             # Upload code + run benchmark
├── teardown.sh           # Delete all Azure resources
├── benchmark-config.json # Default benchmark parameters
└── .env                  # (generated) VM connection info

benchmark.sh              # Multi-configuration benchmark runner
Dockerfile                # GPU container (CUDA 12.1 + dependencies)
requirements-gpu.txt      # CUDA-compatible Python deps
run_poc.py                # Main entry point (single run)
```
