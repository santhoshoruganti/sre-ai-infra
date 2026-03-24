# NVIDIA NCA-AIIO — Study Notes
> **Certification:** NVIDIA-Certified Associate: AI Infrastructure and Operations  


---

## Progress Tracker

- [x] Week 1 — AI Fundamentals
  - [x] AI vs ML vs DL vs GenAI
  - [x] CPU vs GPU — why GPUs win for AI
  - [ ] GPU architecture — CUDA cores, Tensor Cores
  - [ ] Training vs Inference


- [ ] Week 2 — NVIDIA Hardware & Software Stack
  - [ ] GPU generations — A100, H100, B200
  - [ ] DGX systems
  - [ ] NVLink and InfiniBand
  - [ ] NVIDIA software — CUDA, NGC, DCGM, TensorRT


- [ ] Week 3 — AI Infrastructure Design
  - [ ] Data center building blocks
  - [ ] Compute, networking, storage for AI
  - [ ] Reference Architectures (RAs)
  - [ ] On-prem vs Cloud

  
- [ ] Week 4 — AI Operations + Exam Prep
  - [ ] Infrastructure management and monitoring
  - [ ] Cluster orchestration
  - [ ] MIG (Multi-Instance GPU)
  - [ ] Practice questions

---

## Domain 1 — Introduction to AI

### 1.1 AI vs ML vs DL vs GenAI

Think of it as a **Russian nesting doll** — each concept lives inside the previous one.

```
┌─────────────────────────────────────────────────┐
│  Artificial Intelligence (AI)                   │
│  Any technique making machines seem intelligent │
│  ┌───────────────────────────────────────────┐  │
│  │  Machine Learning (ML)                    │  │
│  │  Machines learn patterns from data        │  │
│  │  ┌─────────────────────────────────────┐  │  │
│  │  │  Deep Learning (DL)                 │  │  │
│  │  │  Neural networks with many layers   │  │  │
│  │  │  ┌───────────────────────────────┐  │  │  │
│  │  │  │  Generative AI (GenAI)        │  │  │  │
│  │  │  │  DL trained to CREATE content │  │  │  │
│  │  │  └───────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

| Term | What it is | Real-world example |
|------|-----------|-------------------|
| **AI** | Broad umbrella — machines doing intelligent tasks | Spam filters, chess engines |
| **ML** | AI that learns from data instead of hand-coded rules | Datadog anomaly detection, recommendation systems |
| **Deep Learning** | ML using multi-layer neural networks | Image recognition, speech-to-text |
| **GenAI** | DL models trained to *create* new content | Claude, ChatGPT, GitHub Copilot, Stable Diffusion |

#### Key distinctions to remember for the exam
- **ML** = learning patterns from data (no explicit programming of rules)
- **DL** = subset of ML — uses neural nets with many layers, needs GPUs
- **GenAI** = subset of DL — goal is *generation*, not just classification/prediction
- Not all AI is ML (e.g. rule-based expert systems)
- Not all ML is DL (e.g. Random Forests, SVMs)

#### SRE connection
> ML-based anomaly detection tools (like Datadog AIOps) watch your metrics, learn "normal" patterns, and alert on deviations — instead of you manually setting thresholds. That's ML in your day job.

---

### 1.2 CPU vs GPU — Why GPUs Win for AI

#### The core idea
AI training = doing the **same simple math** (multiply + add) **billions of times in parallel** across huge matrices.  
GPUs are built exactly for this. CPUs are not.

#### The restaurant analogy
| | CPU | GPU |
|--|-----|-----|
| Think of it as | 1 Michelin-star head chef | 4,000 line cooks |
| Strengths | Complex decisions, versatile, fast per-task | Simple tasks done all at once |
| Best for | OS, databases, web servers, general logic | Matrix math, AI training, rendering |
| Weakness | Sequential bottleneck on parallel work | Poor at complex branching logic |

**Task: "Chop 10,000 onions"**  
→ Head chef (CPU): hours  
→ 4,000 line cooks (GPU): minutes

#### Real numbers (exam relevant)

| | Intel Xeon CPU | NVIDIA H100 GPU |
|--|---------------|----------------|
| Cores | 60–120 | 16,896 CUDA cores |
| FP32 performance | ~2 TFLOPS | ~60 TFLOPS |
| AI (Tensor) perf | N/A | ~2,000 TFLOPS |
| Memory bandwidth | ~300 GB/s | 3,350 GB/s |
| Best for | General compute | AI training & inference |

#### Key terms
- **TFLOPS** — Trillion Floating Point Operations Per Second. Measures raw compute power.
- **CUDA cores** — NVIDIA's parallel processing units. Do the actual math.
- **Tensor Cores** — Special hardware inside NVIDIA GPUs. Do a full 4×4 matrix multiply in one clock cycle. This is why AI perf is so much higher than raw FP32 perf.
- **Memory bandwidth** — Speed of moving data to/from GPU memory. Often the real bottleneck, not compute.

#### SRE connection
> - **GPU utilisation %** is your key health metric (not CPU%). An H100 at 40% utilisation = wasted money.
> - **DCGM** (Data Center GPU Manager) is your `htop` for GPUs — shows utilisation, temp, power, memory.
> - Memory bandwidth bottlenecks before compute — watch for it.

---

### 1.3 Training vs Inference

| | Training | Inference |
|--|----------|-----------|
| What happens | Model *learns* from data | Model *uses* what it learned |
| Compute need | Extremely high (weeks of GPU time) | Lower, but still needs GPUs at scale |
| When it runs | Once (or periodically) | Continuously, on every user request |
| Infrastructure | Large GPU clusters (DGX SuperPOD) | GPU servers, sometimes smaller/optimised |
| Cost | Very high | Moderate but ongoing |

> **Exam tip:** Training and inference have very different infrastructure profiles. Training = burst, massive parallelism. Inference = sustained, latency-sensitive.

---

## Domain 2 — NVIDIA Hardware & Software Stack

### 2.1 GPU Generations

| GPU | Generation | Key feature | Use case |
|-----|-----------|-------------|----------|
| A100 | Ampere (2020) | 3rd-gen Tensor Cores, MIG support | Training, HPC |
| H100 | Hopper (2022) | 4th-gen Tensor Cores, Transformer Engine | LLM training/inference |
| B200 | Blackwell (2024) | 2x H100 performance | Next-gen LLM workloads |
| L40S | Ada Lovelace | Balanced compute + graphics | Inference, visualisation |

> **H100 is the most exam-relevant GPU** — it's the current standard for enterprise AI infrastructure.

---

### 2.2 DGX Systems

NVIDIA's **DGX** is a purpose-built AI server — all components optimised together.

| System | GPUs | Use case |
|--------|------|----------|
| DGX H100 | 8× H100 GPUs | Single-node AI training |
| DGX SuperPOD | Multiple DGX nodes + InfiniBand | Large-scale cluster training |
| DGX Cloud | Cloud-hosted DGX | On-demand AI infrastructure |

> Think of DGX as a **pre-validated, NVIDIA-optimised "AI appliance"** — like a Dell PowerEdge but purpose-built for AI.

---

### 2.3 GPU Interconnects

When multiple GPUs work together, they need fast links to share data.

| Technology | What it is | Speed | Use case |
|-----------|-----------|-------|---------|
| **NVLink** | GPU-to-GPU interconnect *within* a server | 900 GB/s | Multi-GPU training on one DGX node |
| **NVSwitch** | Switch chip connecting all GPUs on a node | Full NVLink bandwidth | All-to-all GPU communication |
| **InfiniBand** | High-speed network *between* servers | Up to 400 Gb/s | Multi-node cluster communication |
| **PCIe** | Standard server bus | ~128 GB/s | Fallback — much slower than NVLink |

> **SRE analogy:**  
> NVLink = fast local loopback between processes on same machine  
> InfiniBand = low-latency dedicated backbone between servers  
> PCIe = the general-purpose backplane — works, but not optimised for AI

---

### 2.4 NVIDIA Software Stack

```
┌──────────────────────────────────────────┐
│  AI Applications (GPT, Stable Diffusion) │
├──────────────────────────────────────────┤
│  Frameworks (PyTorch, TensorFlow)        │
├──────────────────────────────────────────┤
│  NVIDIA AI Enterprise / NGC Catalog      │
├──────────────────────────────────────────┤
│  TensorRT / Triton Inference Server      │
├──────────────────────────────────────────┤
│  CUDA (Compute Unified Device Arch.)     │
├──────────────────────────────────────────┤
│  GPU Hardware (H100, A100...)            │
└──────────────────────────────────────────┘
```

| Tool | What it does | SRE equivalent |
|------|-------------|---------------|
| **CUDA** | Programming model for GPU compute | The OS kernel for GPU work |
| **cuDNN** | Deep learning primitives library | libc for neural network ops |
| **TensorRT** | Optimises models for fast inference | JIT compiler / performance tuner |
| **Triton** | Model serving framework | Nginx/Envoy — but for AI models |
| **NGC Catalog** | Pre-built containers for AI workloads | Docker Hub for AI |
| **DCGM** | GPU monitoring and management | Prometheus + node_exporter for GPUs |
| **RAPIDS** | GPU-accelerated data processing | GPU-powered pandas/scikit-learn |
| **NVIDIA AI Enterprise** | Full software suite for enterprise AI | NVIDIA's "enterprise support" stack |

---

## Domain 3 — AI Infrastructure

### 3.1 Data Center Building Blocks

Every AI data center needs three pillars:

```
        COMPUTE          NETWORKING        STORAGE
       ┌────────┐       ┌──────────┐      ┌─────────┐
       │ GPU    │◄─────►│InfiniBand│◄────►│ NVMe /  │
       │servers │       │ switches │      │ GPFS /  │
       │ (DGX) │       │          │      │ Lustre  │
       └────────┘       └──────────┘      └─────────┘
```

| Pillar | Key tech | Why it matters for AI |
|--------|---------|----------------------|
| **Compute** | DGX H100, GPU clusters | Raw training/inference power |
| **Networking** | InfiniBand HDR/NDR, NCCL | GPUs need to share gradients fast |
| **Storage** | NVMe SSDs, parallel FS (GPFS, Lustre) | Feeding training data fast enough |

> **Storage is often the hidden bottleneck** — a GPU cluster starved of training data runs at low utilisation. High-throughput parallel file systems solve this.

---

### 3.2 MIG — Multi-Instance GPU

**MIG (Multi-Instance GPU)** lets you partition a single H100 into up to **7 isolated GPU instances**, each with their own memory and compute.

```
┌─────────────────────────── H100 GPU ──────────────────────────┐
│  MIG 1  │  MIG 2  │  MIG 3  │  MIG 4  │  MIG 5  │  MIG 6  │  │
│ 1/7th   │ 1/7th   │ 1/7th   │ 1/7th   │ 1/7th   │ 1/7th   │  │
│ compute │ compute │ compute │ compute │ compute │ compute │  │
└─────────────────────────────────────────────────────────────┘
```

**Why it matters:**
- Run 7 smaller inference workloads on one GPU instead of wasting a full H100 on one small job
- Each MIG instance is fully isolated — fault in one doesn't affect others
- Enables better GPU utilisation (your key SRE metric!)

> **SRE analogy:** MIG is like CPU cgroups / Kubernetes resource limits — you're slicing one physical resource into isolated, guaranteed partitions.

---

### 3.3 Reference Architectures (RAs)

NVIDIA publishes **Reference Architectures** — validated, best-practice designs for building AI infrastructure.

Think of them like:
- AWS Well-Architected Framework, but for GPU clusters
- Pre-validated blueprints so you don't start from scratch

Examples:
- NVIDIA DGX BasePOD — small AI cluster starting point
- NVIDIA DGX SuperPOD — large-scale production AI cluster
- NVIDIA AI Enterprise RA — enterprise software stack

---

### 3.4 On-Prem vs Cloud AI

| | On-Premises | Cloud (AWS, Azure, GCP) |
|--|-------------|------------------------|
| Cost model | CapEx (buy hardware) | OpEx (pay per use) |
| GPU availability | Fixed — what you buy | Elastic — scale up/down |
| Latency | Low | Depends on region |
| Control | Full | Limited by provider |
| Best for | Steady, predictable workloads | Burst/experimental workloads |
| Examples | DGX SuperPOD in your DC | AWS p4d.24xlarge, Azure NDv4 |

> **SRE note:** Many enterprises use a **hybrid model** — on-prem for steady-state training, cloud for burst capacity.

---

## Domain 4 — AI Operations

### 4.1 Key Monitoring Metrics (DCGM)

As an SRE, this is your home territory. For AI infra, watch these:

| Metric | Tool | What to watch |
|--------|------|--------------|
| GPU utilisation % | DCGM | Should be >80% during training |
| GPU memory used | DCGM | Near 100% is normal and fine |
| GPU temperature | DCGM | Alert above 83°C |
| Power draw (W) | DCGM | Compare against TDP |
| NVLink bandwidth | DCGM | Bottleneck indicator |
| PCIe throughput | DCGM | Low = data starved |
| SM (core) occupancy | DCGM | Low = underutilisation |

> **Key insight:** Unlike CPU servers where 100% utilisation is bad, GPU memory at 100% is *expected and desired* — it means you're using the hardware efficiently.

---

### 4.2 Cluster Orchestration

| Tool | What it does |
|------|-------------|
| **Kubernetes** | Container orchestration — standard for inference workloads |
| **Slurm** | HPC job scheduler — common for training clusters |
| **NVIDIA GPU Operator** | Kubernetes operator that manages GPU drivers and DCGM automatically |
| **NCCL** | NVIDIA Collective Communications Library — manages GPU-to-GPU data exchange during training |

---

### 4.3 Key Networking Concepts

| Concept | What it is |
|---------|-----------|
| **NCCL** | Library coordinating gradient sharing across GPUs during training |
| **GPUDirect** | Allows GPUs to talk directly to network cards (bypassing CPU) — reduces latency |
| **RoCE** | RDMA over Converged Ethernet — InfiniBand-like performance over Ethernet |
| **All-reduce** | Operation where all GPUs share and aggregate gradients — needs high bandwidth |

---

## Exam Quick-Reference

### Key numbers to remember

| Item | Number |
|------|--------|
| H100 CUDA cores | 16,896 |
| H100 Tensor TFLOPS (FP8) | ~3,958 TFLOPS |
| H100 memory bandwidth | 3,350 GB/s |
| Max MIG instances (H100) | 7 |
| NVLink bandwidth (H100) | 900 GB/s |
| DGX H100 GPUs per node | 8 |

### Acronyms cheat sheet

| Acronym | Full form |
|---------|-----------|
| CUDA | Compute Unified Device Architecture |
| DCGM | Data Center GPU Manager |
| MIG | Multi-Instance GPU |
| NCCL | NVIDIA Collective Communications Library |
| NGC | NVIDIA GPU Cloud |
| RA | Reference Architecture |
| TensorRT | Tensor Runtime (inference optimiser) |
| DL | Deep Learning |
| HPC | High Performance Computing |
| NVLink | NVIDIA's GPU-to-GPU interconnect |

---

## Resources

| Resource | Link | Notes |
|----------|------|-------|
| NVIDIA Academy (official course) | https://academy.nvidia.com | ~7 hours, paid |
| Coursera (same course) | https://coursera.org | Free audit available |
| FreeCodeCamp prep course | https://youtube.com/freecodecamp | Free, 4 hours, March 2026 |
| NVIDIA DCGM docs | https://docs.nvidia.com/datacenter/dcgm | SRE-relevant |
| NGC Catalog | https://catalog.ngc.nvidia.com | Browse AI containers |

---

*Notes by Santhosh — SRE → AI Infrastructure Engineer path*  
*Last updated: March 2026*
