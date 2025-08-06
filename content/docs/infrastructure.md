---
title: Module 3 - Infrastructure & Deployment
weight: 4
---

# Infrastructure & Deployment Module

Leverage your infrastructure expertise to build scalable, reliable ML systems.

## Learning Objectives

- Design ML-specific infrastructure patterns
- Implement containerized ML deployments
- Orchestrate ML workloads on Kubernetes
- Optimize model serving for production

## Week 9: Containerization for ML

### Docker for ML Applications

#### Multi-stage Build Pattern
```dockerfile
# Build stage
FROM python:3.9 AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "serve.py"]
```

### GPU Support
```dockerfile
FROM nvidia/cuda:11.8.0-base-ubuntu22.04
# Install Python and ML frameworks
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Container Optimization
- Layer caching strategies
- Size reduction techniques
- Security scanning
- Registry management

## Week 10: Kubernetes for ML

### ML Workload Patterns

#### Training Jobs
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: ml-training:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
      restartPolicy: OnFailure
```

#### Model Serving
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: model
        image: model-serve:latest
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
```

### Kubeflow Components
- Pipelines for orchestration
- KFServing for model serving
- Katib for hyperparameter tuning
- Notebooks for development

## Week 11: Model Serving Architectures

### Serving Frameworks Comparison

| Framework | Use Case | Pros | Cons |
|-----------|----------|------|------|
| TensorFlow Serving | TF models | High performance | TF-specific |
| TorchServe | PyTorch models | Easy deployment | PyTorch only |
| Triton | Multi-framework | GPU optimization | Complex setup |
| Seldon Core | Any framework | K8s native | Overhead |

### Implementation Examples

#### FastAPI Model Server
```python
from fastapi import FastAPI
import torch
import numpy as np

app = FastAPI()
model = torch.load("model.pt")

@app.post("/predict")
async def predict(data: dict):
    input_tensor = torch.tensor(data["features"])
    with torch.no_grad():
        prediction = model(input_tensor)
    return {"prediction": prediction.tolist()}
```

#### gRPC for Low Latency
```python
import grpc
from concurrent import futures
import model_pb2
import model_pb2_grpc

class ModelService(model_pb2_grpc.ModelServicer):
    def Predict(self, request, context):
        features = np.array(request.features)
        prediction = self.model.predict(features)
        return model_pb2.PredictionResponse(
            predictions=prediction.tolist()
        )
```

### Edge Deployment
- Model quantization
- ONNX conversion
- TensorFlow Lite
- Core ML for iOS

## Week 12: Infrastructure as Code

### Terraform for ML Infrastructure

```hcl
# GPU-enabled training cluster
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.ml_cluster.name
  node_group_name = "gpu-nodes"
  
  instance_types = ["p3.2xlarge"]
  
  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 1
  }
  
  labels = {
    workload = "ml-training"
    gpu      = "true"
  }
}

# Model storage
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "ml-model-artifacts"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    enabled = true
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}
```

### Cost Optimization
- Spot instances for training
- Auto-scaling policies
- Resource tagging
- Cost monitoring dashboards

## Hands-on Projects

### Project 1: Production ML Service
Build a complete ML service with:
- Containerized model server
- Kubernetes deployment
- Auto-scaling based on metrics
- Blue-green deployment
- Monitoring and logging

### Project 2: Multi-Cloud ML Platform
Design infrastructure supporting:
- Training on AWS/GCP
- Model registry on S3
- Serving on edge devices
- Cost optimization

## Performance Optimization

### Model Optimization Techniques
- Quantization (INT8, FP16)
- Pruning unused weights
- Knowledge distillation
- Batch inference

### Infrastructure Optimization
- GPU utilization monitoring
- Memory optimization
- Network bandwidth management
- Caching strategies

## Assessment

### Practical Tasks
- [ ] Containerize an ML application with GPU support
- [ ] Deploy model on Kubernetes with auto-scaling
- [ ] Implement A/B testing for models
- [ ] Create IaC for ML infrastructure
- [ ] Optimize model serving latency

### Performance Metrics
- Model serving latency < 100ms
- 99.9% availability SLA
- Cost per inference < $0.001
- GPU utilization > 80%

## Tools & Resources

### Essential Tools
- **Containers**: Docker, Podman
- **Orchestration**: Kubernetes, EKS, GKE
- **IaC**: Terraform, Pulumi
- **Serving**: TF Serving, TorchServe, Triton
- **Monitoring**: Prometheus, Grafana

### Learning Resources
- [Kubernetes Patterns for ML](https://kubernetes.io/blog)
- [NVIDIA Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server)
- [ML Infrastructure Best Practices](https://ml-ops.org)

## Next Steps

Proceed to [Module 4: Data Engineering](/docs/data-engineering) to master data pipeline design and feature engineering for ML systems.