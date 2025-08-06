---
title: Development Environment Setup
weight: 10
---

# Development Environment Setup

Complete guide to set up your ML engineering development environment.

## System Requirements

### Hardware
- **Minimum**: 8GB RAM, 50GB storage, 4 CPU cores
- **Recommended**: 16GB RAM, 200GB SSD, 8 CPU cores
- **GPU** (optional): NVIDIA GPU with CUDA support for deep learning

### Operating System
- Ubuntu 20.04+ / macOS 11+ / Windows 11 with WSL2
- Docker Desktop installed
- Kubernetes (minikube or kind for local development)

## Core Software Installation

### 1. Python Environment

```bash
# Install Python 3.9+
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip

# Install Conda (alternative)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create virtual environment
python3.9 -m venv ml-env
source ml-env/bin/activate
```

### 2. Essential Python Packages

```bash
# Core ML libraries
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow pytorch torchvision
pip install xgboost lightgbm

# MLOps tools
pip install mlflow dvc wandb
pip install prefect airflow
pip install pytest black flake8

# Serving frameworks
pip install fastapi uvicorn gunicorn
pip install streamlit gradio
```

### 3. Docker Setup

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 4. Kubernetes Tools

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install minikube for local development
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

## IDE Configuration

### VS Code Extensions
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-toolsai.jupyter",
    "ms-azuretools.vscode-docker",
    "ms-kubernetes-tools.vscode-kubernetes-tools",
    "GitHub.copilot",
    "ms-vscode-remote.remote-containers"
  ]
}
```

### PyCharm Setup
1. Install PyCharm Professional
2. Configure Python interpreter
3. Install plugins: Docker, Kubernetes, Database Tools
4. Set up code style and linting

## Cloud Platform CLIs

### AWS
```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure credentials
aws configure
```

### Google Cloud
```bash
# Install gcloud SDK
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt update && sudo apt install google-cloud-cli
```

### Azure
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login
```

## GPU Setup (Optional)

### NVIDIA CUDA Installation
```bash
# Check GPU
lspci | grep -i nvidia

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda

# Install cuDNN
# Download from NVIDIA website and install

# Verify installation
nvidia-smi
nvcc --version
```

### PyTorch with CUDA
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Project Structure Template

```
ml-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── models/
│   ├── checkpoints/
│   └── production/
├── notebooks/
│   ├── exploration/
│   └── experiments/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── serving/
├── tests/
│   ├── unit/
│   └── integration/
├── configs/
│   ├── training/
│   └── deployment/
├── docker/
│   ├── training.Dockerfile
│   └── serving.Dockerfile
├── kubernetes/
│   ├── training/
│   └── serving/
├── .github/
│   └── workflows/
├── requirements.txt
├── setup.py
├── Makefile
└── README.md
```

## Verification Script

Create a script to verify your setup:

```python
#!/usr/bin/env python3
"""Verify ML development environment setup."""

import subprocess
import sys

def check_command(cmd, name):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {name} is installed")
            return True
    except:
        pass
    print(f"✗ {name} is not installed")
    return False

def main():
    checks = [
        ("python3 --version", "Python 3"),
        ("docker --version", "Docker"),
        ("kubectl version --client", "kubectl"),
        ("git --version", "Git"),
        ("aws --version", "AWS CLI"),
    ]
    
    print("Checking environment setup...\n")
    all_good = all(check_command(cmd, name) for cmd, name in checks)
    
    print("\n" + "="*40)
    if all_good:
        print("✓ All checks passed! Environment is ready.")
    else:
        print("✗ Some components are missing. Please install them.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **Permission Denied for Docker**
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

2. **Python Package Conflicts**
   ```bash
   # Use virtual environments
   python -m venv fresh-env
   source fresh-env/bin/activate
   ```

3. **CUDA Version Mismatch**
   - Check compatibility matrix for PyTorch/TensorFlow
   - Use Docker containers with pre-configured CUDA

## Next Steps

1. Clone the course repository
2. Run the verification script
3. Complete the [Module 1: Fundamentals](/docs/fundamentals)
4. Join the community Discord/Slack channel