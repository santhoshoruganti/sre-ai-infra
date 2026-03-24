# CUDA Practice Environments — No Cloud Budget Needed

> How to practice GPU and CUDA tasks as an SRE without real Azure/AWS infrastructure.
> Covers Google Colab, self-installation, and GitHub Codespaces / Play with Docker.

---

## Table of Contents

- [Google Colab](#1-google-colab)
- [Self Installation (Local)](#2-self-installation-local)
- [GitHub Codespaces](#3-github-codespaces)
- [Play with Docker](#4-play-with-docker)
- [Practice Exercise Checklist](#practice-exercise-checklist)

---

# 1. Google Colab

**Best for:** Practicing real GPU commands instantly — zero setup, free.

## Setup

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **Runtime** → **Change runtime type**
3. Select **T4 GPU** → Save
4. You now have a real NVIDIA T4 GPU to practice on

## What You Can Practice

### Check GPU status
```bash
!nvidia-smi
```

### Check CUDA toolkit and driver version
```bash
!nvcc --version
!cat /usr/local/cuda/version.txt
!nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

### Check GPU memory and temperature
```bash
!nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits
```

### Live GPU monitoring (5 readings)
```bash
!nvidia-smi dmon -s um -d 1 -c 5
```

### Simulate a GPU workload and watch utilisation spike
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(f"CUDA version: {torch.version.cuda}")

a = torch.randn(10000, 10000).cuda()
b = torch.randn(10000, 10000).cuda()
result = torch.matmul(a, b)
print("Done — run nvidia-smi to see memory used")
```

### Check GPU memory from Python
```python
import torch
print(torch.cuda.memory_summary())
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Simulate OOM (Out of Memory) — practice incident scenario
```python
import torch
tensors = []
try:
    while True:
        tensors.append(torch.randn(1000, 1000).cuda())
except RuntimeError as e:
    print(f"OOM hit: {e}")
```

### Recover from OOM — clear GPU memory
```python
import torch, gc
tensors = []
gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
```

## Tips for Colab

- Disconnects after ~90 mins idle — keep a cell running to stay active
- Use `!` before shell commands (e.g. `!nvidia-smi`)
- Use `%%bash` at top of a cell to run the whole cell as bash
- Save notebooks to GitHub so you don't lose your practice scripts


---

# 2. Self Installation (Local)

**Best for:** Practicing Kubernetes GPU scheduling, Docker CUDA images, toolkit commands — no real GPU needed.

## 2a. Install CUDA Toolkit (Ubuntu / WSL2 on Windows)

Even without an NVIDIA GPU you can install the toolkit and learn the tooling.

```bash
# Add NVIDIA package repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-2

# Verify
nvcc --version
ls /usr/local/cuda/
```

### Set PATH (add to ~/.bashrc)
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

### What you can explore without a GPU
```bash
nvcc --version                      # CUDA compiler
ls /usr/local/cuda/include/         # CUDA headers
ls /usr/local/cuda/lib64/           # cuBLAS, cuDNN, NCCL libraries
cat /usr/local/cuda/version.json    # Full version info
```

---

## 2b. Local Kubernetes with Kind

Practice all kubectl GPU commands without any real GPU.

### Install Kind + kubectl
```bash
# Kind — Linux / WSL2
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind

# kubectl — Linux / WSL2
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Mac — both via Homebrew
brew install kind kubectl
```

### Create local cluster
```bash
kind create cluster --name gpu-practice
kubectl cluster-info
kubectl get nodes
```

### Install NVIDIA device plugin
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/deployments/static/nvidia-device-plugin.yml
kubectl get pods -n kube-system | grep nvidia
```

### Practice GPU pod scheduling
```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: cuda-container
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1
    command: ["nvidia-smi"]

---

# 3. GitHub Codespaces

**Best for:** Pre-configured cloud environment in your browser — no local install needed.

> Note: Codespaces does NOT provide a real GPU — but you can practice all Kubernetes, Docker, and CUDA toolkit commands perfectly.

## Setup

1. Go to [github.com](https://github.com) → your repo
2. Click **Code** → **Codespaces** → **Create codespace on main**
3. You get a full VS Code environment in the browser with terminal

## What to Practice

### Install CUDA toolkit in Codespaces terminal
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-2
nvcc --version
```

### Install Kind + kubectl and create a cluster
```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

kind create cluster --name gpu-lab
kubectl get nodes
```

### Practice writing GPU pod manifests
```bash
cat <<EOF > gpu-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  containers:
  - name: trainer
    image: nvidia/cuda:12.2.0-runtime-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 2
    env:
    - name: NCCL_DEBUG
      value: "INFO"

---

# 3. GitHub Codespaces

**Best for:** Pre-configured cloud environment in your browser — no local install needed.

> Note: Codespaces does NOT give a real GPU — but you can practice all Kubernetes, Docker, and CUDA toolkit commands.

## Setup

1. Go to [github.com](https://github.com) → your repo
2. Click **Code** → **Codespaces** → **Create codespace on main**
3. You get VS Code in the browser with a full terminal

## What to Practice

### Install CUDA toolkit
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-2
nvcc --version
```

### Install Kind + kubectl and create cluster
```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind && sudo mv ./kind /usr/local/bin/kind
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
kind create cluster --name gpu-lab
kubectl get nodes
```

### Write and apply a GPU pod manifest
```bash
cat <<YAML > gpu-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  containers:
  - name: trainer
    image: nvidia/cuda:12.2.0-runtime-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 2
    env:
    - name: NCCL_DEBUG
      value: "INFO"
YAML
kubectl apply -f gpu-pod.yaml
kubectl describe pod gpu-workload
kubectl delete pod gpu-workload
```

### Install Helm and dry-run dcgm-exporter
```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm repo update
helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace monitoring --create-namespace --dry-run
```

---

# 4. Play with Docker

**Best for:** Quick Docker experiments — fully in browser, no install, free 4-hour sessions.

## Setup

1. Go to [labs.play-with-docker.com](https://labs.play-with-docker.com)
2. Login with Docker Hub account (free at hub.docker.com)
3. Click **+ ADD NEW INSTANCE** — terminal ready in seconds

## What to Practice

### Pull and explore CUDA images
```bash
docker pull nvidia/cuda:12.2.0-base-ubuntu22.04
docker images
docker run -it nvidia/cuda:12.2.0-base-ubuntu22.04 bash
```

### Inside the container
```bash
ls /usr/local/cuda/
cat /usr/local/cuda/version.txt
env | grep -i cuda
ls /usr/local/cuda/lib64/ | head -20
```

### Compare image sizes — base vs runtime vs devel
```bash
docker pull nvidia/cuda:12.2.0-base-ubuntu22.04
docker pull nvidia/cuda:12.2.0-runtime-ubuntu22.04
docker pull nvidia/cuda:12.2.0-devel-ubuntu22.04
docker images | grep cuda
# base ~200MB, runtime ~1.5GB, devel ~4GB+
```

### Inspect image layers
```bash
docker history nvidia/cuda:12.2.0-base-ubuntu22.04
docker inspect nvidia/cuda:12.2.0-base-ubuntu22.04 | grep -A5 Env
```

### Build a custom CUDA image
```bash
cat > Dockerfile << 'FILE'
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3
CMD ["python3", "--version"]
FILE
docker build -t my-cuda-app .
docker run my-cuda-app
```

---

# Practice Exercise Checklist

## Google Colab
- [ ] Run `nvidia-smi` and read every field correctly
- [ ] Identify: GPU name, driver version, CUDA version, total memory
- [ ] Run matrix multiplication and watch utilisation spike
- [ ] Trigger an OOM error intentionally
- [ ] Recover from OOM using `torch.cuda.empty_cache()`
- [ ] Check memory with `torch.cuda.memory_summary()`

## Local / Codespaces — CUDA Toolkit
- [ ] Install CUDA toolkit successfully
- [ ] Run `nvcc --version`
- [ ] Set PATH and LD_LIBRARY_PATH in ~/.bashrc
- [ ] Explore `/usr/local/cuda/lib64/` — find cuDNN, cuBLAS

## Local / Codespaces — Kubernetes
- [ ] Create a Kind cluster
- [ ] Install NVIDIA device plugin
- [ ] Write a pod spec requesting 1 GPU
- [ ] Practice `kubectl cordon` → `drain` → `uncordon`
- [ ] Check node GPU capacity with `kubectl describe node`
- [ ] Dry-run dcgm-exporter Helm install

## Docker / Play with Docker
- [ ] Pull `nvidia/cuda:12.2.0-base-ubuntu22.04`
- [ ] Compare base vs runtime vs devel image sizes
- [ ] Explore `/usr/local/cuda/` inside a container
- [ ] Build a custom Dockerfile from a CUDA base image
- [ ] Use `docker history` to inspect image layers

---

## Suggested Learning Order

```
Week 1 → Google Colab        — Real GPU, browser only, zero setup
                                nvidia-smi, GPU memory, OOM practice

Week 2 → Play with Docker    — Browser only, zero setup
                                CUDA images, Dockerfiles, image sizes

Week 3 → GitHub Codespaces   — Kind cluster, kubectl, Helm, node drain

Week 4 → Combine all three   — Write a full OOM incident runbook
```
