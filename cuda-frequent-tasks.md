# CUDA — Frequent Tasks & What to Know

> Practical day-to-day reference for SRE and AI Infra engineers working with CUDA and GPU infrastructure.
> Split by role — jump to your section.

---

## 📌 Table of Contents

- [SRE Engineer — Frequent Tasks](#sre-engineer--frequent-tasks)
- [SRE Engineer — What You Should Be Familiar With](#sre-engineer--what-you-should-be-familiar-with)
- [AI Infra Engineer — Frequent Tasks](#ai-infra-engineer--frequent-tasks)
- [AI Infra Engineer — What You Should Be Familiar With](#ai-infra-engineer--what-you-should-be-familiar-with)

---

# 👷 SRE Engineer — Frequent Tasks

## Daily Tasks

**1. Check GPU fleet health**
```bash
nvidia-smi                          # Quick status of all GPUs
nvidia-smi dmon -s um -d 1         # Live utilisation + memory stream
```
- Look for: high temperature (>85°C), ECC errors, GPUs stuck at 0% util during active jobs

**2. Monitor alerts**
- Check Grafana dashboards for `DCGM_FI_DEV_XID_ERRORS`, `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_FB_USED`
- Any XID error alert = treat as potential hardware fault, investigate immediately

**3. Check Kubernetes GPU node status**
```bash
kubectl get nodes -l accelerator=nvidia    # Are GPU nodes Ready?
kubectl describe node <node> | grep -A5 Allocatable   # How many GPUs free?
```

---

## Weekly Tasks

**4. Run GPU diagnostics on new or suspect nodes**
```bash
dcgmi diag -r 1       # Quick check (~2 mins)
dcgmi diag -r 3       # Full check (~10 mins) — run before admitting new nodes
```

**5. Review GPU utilisation trends**
- Are GPUs consistently under 30% utilisation? That's a cost problem — flag to AI Infra team
- Are jobs failing due to OOM regularly? Check if capacity needs increasing

**6. Check driver versions across fleet**
```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
```
- All nodes in same pool should run the same driver version
- Mismatches cause hard-to-debug failures

---

## On-Demand / Incident Tasks

**7. Handle OOM (Out of GPU Memory) incident**
```bash
nvidia-smi                          # Find which process is eating memory
nvidia-smi pmon                     # Per-process breakdown
kill -9 <pid>                       # Kill stuck process if needed
```
- After recovery: check what model/batch size caused it, flag to AI Infra team

**8. Diagnose driver/toolkit mismatch**
```bash
# Error seen: "CUDA driver version is insufficient for CUDA runtime version"
nvidia-smi | head -3                # Check driver version on node
# Compare with CUDA version in the container image being used
# Fix: upgrade node driver OR ask team to use older CUDA base image
```

**9. Handle XID hardware error**
```bash
dmesg | grep -i xid                 # See what XID code fired
nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv
```

| XID | Meaning | Action |
|-----|---------|--------|
| 48 | Double-bit ECC error | Drain node, raise hardware ticket |
| 79 | GPU hung | Reboot node |
| 92 | High single-bit ECC | Monitor closely, plan replacement |
| 94 | Contained channel error | Drain node, check with vendor |

**10. GPU not visible in pod**
```bash
kubectl get pods -n kube-system | grep nvidia-device-plugin   # Is plugin running?
kubectl describe pod <pod> | grep -i gpu                      # Is GPU requested correctly?
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi  # Test on node
```

**11. Node maintenance — drain GPU node safely**
```bash
kubectl cordon <node>               # Stop new pods scheduling here
kubectl drain <node> --ignore-daemonsets --delete-emptydir-data
# Do your maintenance (driver upgrade, reboot, etc.)
kubectl uncordon <node>             # Re-admit to cluster
```

---

## Ongoing Responsibilities

**12. Keep NVIDIA device plugin up to date**
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml
```

**13. Maintain dcgm-exporter for Prometheus metrics**
```bash
helm upgrade dcgm-exporter gpu-helm-charts/dcgm-exporter -n monitoring
```

**14. Capacity planning**
- Track `DCGM_FI_DEV_FB_USED` (VRAM usage) trends over time
- Track GPU node pool utilisation weekly
- Flag to management when fleet is consistently >80% booked

---

## 🧠 SRE Engineer — What You Should Be Familiar With

### Must Know
- **nvidia-smi** — read every field, understand what SM utilisation, framebuffer, and power draw mean
- **CUDA driver vs toolkit version** — the difference, how compatibility works, how to check both
- **XID error codes** — which ones are urgent hardware faults vs recoverable errors
- **NVIDIA device plugin** — how it works in Kubernetes, how GPUs are exposed to pods
- **dcgm-exporter** — what metrics it exports, how to set up alerts in Prometheus/Grafana
- **GPU node drain procedure** — how to safely remove a GPU node from cluster without killing jobs

### Good to Know
- **ECC memory** — what correctable vs uncorrectable errors mean for GPU health
- **NVLink topology** — `nvidia-smi topo -m` output, what P2P vs SYS means for performance
- **CUDA container runtime** — how `--gpus all` works in Docker, what nvidia-container-runtime does
- **GPU memory is not managed by Kubernetes** — you cannot set memory limits like CPU/RAM; CUDA owns it

### Awareness Level (Don't need to implement, but understand the concept)
- What cuDNN, cuBLAS, NCCL libraries do — so you understand what AI workloads depend on
- What OOM means for model training — why batch size and model size affect VRAM
- What distributed training looks like — why NCCL errors appear in your logs during multi-GPU jobs

---
---

# 🤖 AI Infra Engineer — Frequent Tasks

## Daily Tasks

**1. Check GPU utilisation of running jobs**
```bash
nvidia-smi dmon -s um -d 1         # Live SM util + memory per GPU
```
- SM util < 30% during training = something is wrong — investigate

**2. Support ML engineers with CUDA errors**
- Common ones: OOM errors, CUDA version mismatches, NCCL timeouts
- First question always: what base image are they using? What CUDA version?

**3. Monitor inference serving GPU metrics**
- Track latency, throughput, GPU memory usage for deployed models
- Alert if memory fragmentation is causing slowdowns

---

## Weekly Tasks

**4. Review base image versions**
- Are teams using pinned CUDA versions? No `:latest` tags
- Is there a new cuDNN or toolkit version that improves performance?

**5. Profile at least one training or inference job**
```bash
# Quick check — is the GPU actually busy?
nvidia-smi dmon -s um -d 1

# Deeper — PyTorch profiler
python3 -c "
import torch
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.GPU]) as prof:
    pass  # your model forward pass here
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
"
```

**6. Check NCCL health for distributed training jobs**
```bash
# Set in environment to get NCCL debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
# Re-run the job and check logs for warnings
```

---

## On-Demand / Incident Tasks

**7. Diagnose low GPU utilisation**

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Low SM util + low memory | CPU data pipeline too slow | Increase DataLoader workers, use prefetch |
| High memory + low SM util | Batch size too small | Increase batch size |
| Spiky SM util | CPU/GPU sync overhead | Profile with torch.profiler, reduce sync points |
| Memory full + low util | Memory leak / fragmentation | Restart job, check for tensor accumulation |

**8. Debug CUDA OOM in training**
```bash
# In PyTorch — check what's in GPU memory
import torch
print(torch.cuda.memory_summary())

# Clear cache between runs
torch.cuda.empty_cache()
```
- Check: model size vs GPU VRAM, batch size, gradient accumulation settings
- Rule of thumb: model weights + activations + gradients ≈ 3-4x model size in VRAM

**9. Fix CUDA version mismatch in Docker image**
```dockerfile
# Always pin explicitly — never use :latest
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# runtime  = for inference (smaller)
# devel    = for compiling custom ops (larger, ~4GB+)
# base     = minimal, just CUDA
```

**10. Debug NCCL multi-GPU communication failure**
```bash
# Symptoms: training hangs, NCCL timeout errors
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0      # Force specific network interface
export NCCL_IB_DISABLE=1           # Disable InfiniBand if causing issues (fallback to TCP)

# Check GPU topology
nvidia-smi topo -m
# NV# connections = NVLink (fast)
# SYS = going through system memory (slow, may cause NCCL issues)
```

**11. Set up vLLM for LLM inference**
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b \
  --tensor-parallel-size 2 \         # Split across 2 GPUs
  --gpu-memory-utilization 0.90      # Use 90% of VRAM
```

**12. Profile with Nsight Systems (deep dive)**
```bash
nsys profile -o report --trace cuda,nvtx python train.py
# Open report in Nsight UI to see kernel timeline
```

---

## Ongoing Responsibilities

**13. Own the CUDA base image strategy**
- Decide which CUDA versions are supported
- Maintain a registry of approved base images
- Document upgrade paths when CUDA versions change

**14. Document GPU memory requirements for models**
- Keep a reference: model → minimum VRAM needed
- Helps SRE team with capacity planning

**15. Write runbooks for common GPU failures**
- OOM → what to check, how to fix
- NCCL timeout → step-by-step debug
- Share with SRE team so they can handle tier-1 incidents

---

## 🧠 AI Infra Engineer — What You Should Be Familiar With

### Must Know
- **CUDA toolkit versions and compatibility** — which PyTorch version needs which CUDA version
- **nvidia/cuda Docker image tags** — what `runtime` vs `devel` vs `base` means, when to use each
- **GPU memory (VRAM) math** — how to estimate VRAM needed for a model (weights + activations + gradients)
- **torch.profiler** — how to profile GPU kernels, how to read the output
- **NCCL** — what it does, how to enable debug logging, common failure modes
- **vLLM / TensorRT-LLM** — how GPU memory utilisation and tensor parallelism settings work

### Good to Know
- **NVLink vs PCIe vs InfiniBand** — when each is used, performance implications
- **Tensor parallelism vs data parallelism** — when each is appropriate, how they affect GPU memory
- **PagedAttention** (vLLM) — why it reduces memory fragmentation for LLM inference
- **cuDNN / cuBLAS** — what they do under the hood, why framework versions matter
- **Mixed precision training** — FP16 vs BF16 vs FP32, how it affects VRAM and speed

### Awareness Level (Understand the concept, not the implementation)
- How CUDA kernels are launched — grids, blocks, threads — so you understand profiler output
- What SM (Streaming Multiprocessor) utilisation means — why 100% isn't always the goal
- How the NVIDIA device plugin works in Kubernetes — so you can help debug scheduling issues
- ECC memory errors — what they signal about hardware health

---

## 📋 Quick Task Frequency Summary

| Task | SRE | AI Infra |
|------|-----|----------|
| Check GPU health / nvidia-smi | Daily | Occasionally |
| Monitor Grafana GPU dashboards | Daily | Weekly |
| Handle OOM incident | On-demand | On-demand |
| Debug low GPU utilisation | Rarely | Weekly |
| Profile training jobs | Never | Weekly |
| Manage CUDA base images | Never | Ongoing |
| Driver upgrades | Monthly | Never |
| NCCL debugging | Rarely | On-demand |
| Kubernetes GPU node ops | Daily | Occasionally |
| Capacity planning | Monthly | Input provider |

