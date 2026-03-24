# CUDA — Roles & Responsibilities

> Quick reference: what SRE engineers and AI Infra engineers own when working with CUDA and GPU infrastructure.

---

## SRE Engineer

The SRE owns the **platform layer** — keeping GPU nodes healthy, drivers up to date, and workloads running reliably.

### Core Responsibilities

**Driver & Runtime Management**
- Install and upgrade NVIDIA drivers on GPU nodes
- Ensure CUDA driver version is compatible with toolkit versions used by teams
- Manage driver rollouts across the fleet (canary → full rollout)
- Handle driver version pinning in Kubernetes node pools

**GPU Health Monitoring**
- Set up `dcgm-exporter` to expose GPU metrics to Prometheus
- Build Grafana dashboards for temperature, memory, utilisation, power draw
- Alert on XID hardware errors, ECC uncorrected errors, GPU hung states
- Run `dcgmi diag` checks on new nodes before admitting to the cluster

**Cluster & Kubernetes Operations**
- Install and maintain the NVIDIA device plugin on Kubernetes
- Ensure GPU nodes are correctly labelled and schedulable
- Drain and cordon GPU nodes for maintenance without disrupting running jobs
- Manage GPU node pools — scaling, replacement, OS upgrades

**Incident Response**
- Diagnose and resolve OOM (out of memory) crashes
- Identify stuck/zombie GPU processes and recover nodes
- Investigate XID errors and escalate hardware faults to cloud/vendor
- Handle driver/toolkit version mismatch errors reported by teams

**Capacity Planning**
- Track GPU utilisation trends across the fleet
- Forecast GPU capacity needs based on workload growth
- Coordinate GPU node procurement or cloud instance scaling

---

## AI Infra Engineer

The AI Infra engineer owns the **efficiency and software layer** — making GPUs work well for model training and inference workloads.

### Core Responsibilities

**CUDA Environment Management**
- Own CUDA toolkit version strategy across teams (what version, when to upgrade)
- Maintain base Docker images with pinned CUDA + cuDNN versions
- Ensure framework versions (PyTorch, TensorFlow) are compatible with CUDA toolkit
- Document and enforce image tagging standards (no `:latest`)

**GPU Utilisation & Performance**
- Profile training and inference jobs to identify GPU bottlenecks
- Diagnose low GPU utilisation (CPU bottleneck, small batch size, sync overhead)
- Use `nsys`, `ncu`, and `torch.profiler` to find inefficiencies
- Tune DataLoader workers, batch sizes, and prefetch settings

**Model Serving Infrastructure**
- Deploy and maintain model serving stacks (vLLM, TensorRT-LLM, Triton)
- Configure GPU memory allocation per model to maximise throughput
- Set up multi-GPU inference (tensor parallelism) for large models
- Monitor inference latency and GPU memory fragmentation

**Distributed Training Support**
- Configure NCCL for multi-GPU and multi-node training jobs
- Validate NVLink / InfiniBand topology for training clusters
- Debug collective communication failures (NCCL timeouts, hangs)
- Set `NCCL_DEBUG=INFO` and interpret logs for communication issues

**Developer Support**
- Help ML engineers pick the right CUDA base image for their use case
- Guide teams on GPU memory sizing for new models
- Document common CUDA errors and fixes for the wider engineering team
- Review Dockerfiles and training configs for GPU efficiency

---

## Quick Comparison

| Area | SRE Engineer | AI Infra Engineer |
|------|-------------|-------------------|
| **Drivers** | Installs, upgrades, manages | Consumes, reports mismatches |
| **Monitoring** | Builds dashboards and alerts | Uses metrics to tune workloads |
| **OOM incidents** | Recovers the node | Fixes the code/config causing it |
| **CUDA versions** | Ensures driver compatibility | Owns toolkit + image strategy |
| **GPU utilisation** | Flags low utilisation trends | Diagnoses and fixes root cause |
| **Kubernetes** | Manages GPU node pool | Writes GPU resource requests |
| **Incidents** | First responder | Domain expert for AI workloads |
| **Multi-GPU** | Network/hardware layer | NCCL + parallelism config |

---

## Key Tools by Role

| Tool | SRE | AI Infra |
|------|-----|----------|
| `nvidia-smi` | Daily use | Occasional |
| `dcgmi diag` | Regular health checks | Rarely |
| `dcgm-exporter` | Owns setup | Consumes metrics |
| `nsys` / `ncu` | Rarely | Regular profiling |
| `torch.profiler` | Never | Regular use |
| NCCL debug logs | Rarely | Regular use |
| Helm / kubectl | Daily use | Occasional |
| Docker / base images | Consumes | Owns and maintains |

