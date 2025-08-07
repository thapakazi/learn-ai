---
title: Complete Syllabus
weight: 1
---

# ML Engineering Syllabus for DevOps/SRE Professionals

## Module 1: Foundations (Weeks 1-4)

### [Week 1: Python for ML Engineering](/docs/fundamentals/week1/)
- [Python Basics for DevOps/SRE](/docs/fundamentals/week1/python-basics/)
- [NumPy for Infrastructure Metrics](/docs/fundamentals/week1/numpy-arrays/)
- [Pandas for Log Analysis](/docs/fundamentals/week1/pandas-dataframes/)
- [Matplotlib for Monitoring Dashboards](/docs/fundamentals/week1/matplotlib-visualization/)

### Week 2: Mathematics & Statistics Refresher
- Linear algebra essentials
- Probability and statistics
- Calculus for ML (gradients, optimization)
- Practical applications in ML

### Week 3: Machine Learning Fundamentals
- Supervised vs unsupervised learning
- Classification and regression
- Model evaluation metrics
- Overfitting and regularization
- Cross-validation techniques

### Week 4: Deep Learning Basics
- Neural network architecture
- Backpropagation and gradient descent
- Introduction to TensorFlow/PyTorch
- CNNs and RNNs overview

## Module 2: MLOps Core (Weeks 5-8)

### Week 5: Version Control for ML
- Data versioning with DVC
- Model versioning strategies
- Experiment tracking with MLflow/Weights & Biases
- Git workflows for ML projects

### Week 6: ML Pipeline Orchestration
- Apache Airflow for ML workflows
- Kubeflow Pipelines
- Prefect/Dagster alternatives
- Pipeline monitoring and alerting

### Week 7: CI/CD for ML
- Testing ML code and models
- Automated model validation
- Progressive deployment strategies
- A/B testing for models
- Shadow deployments

### Week 8: Model Registry & Governance
- Model registry patterns
- Model metadata management
- Compliance and audit trails
- Model approval workflows

## Module 3: Infrastructure & Deployment (Weeks 9-12)

### Week 9: Containerization for ML
- Docker for ML applications
- Multi-stage builds for optimization
- GPU support in containers
- Container registries for ML

### Week 10: Kubernetes for ML
- Kubernetes fundamentals review
- Kubeflow deployment
- GPU scheduling and management
- Auto-scaling ML workloads
- Service mesh for ML services

### Week 11: Model Serving
- REST vs gRPC for model serving
- TensorFlow Serving
- TorchServe
- ONNX Runtime
- Triton Inference Server
- Edge deployment considerations

### Week 12: Infrastructure as Code for ML
- Terraform for ML infrastructure
- Pulumi alternatives
- Cost optimization strategies
- Multi-cloud considerations

## Module 4: Data Engineering for ML (Weeks 13-16)

### Week 13: Data Pipeline Architecture
- Batch vs streaming data
- Apache Kafka for ML
- Apache Spark for preprocessing
- Data lake vs data warehouse

### Week 14: Feature Engineering & Stores
- Feature engineering best practices
- Feature stores (Feast, Tecton)
- Feature versioning
- Online vs offline features

### Week 15: Data Quality & Validation
- Data quality monitoring
- Schema validation
- Data drift detection
- Great Expectations framework

### Week 16: ETL/ELT for ML
- Building robust data pipelines
- Apache Beam
- DBT for ML
- Real-time feature computation

## Module 5: Monitoring & Reliability (Weeks 17-20)

### Week 17: Model Monitoring
- Performance metrics tracking
- Model drift detection
- Data drift vs concept drift
- Alerting strategies

### Week 18: Observability for ML
- Distributed tracing for ML
- Prometheus & Grafana for ML
- Custom metrics and dashboards
- Log aggregation patterns

### Week 19: ML System Reliability
- SLIs/SLOs/SLAs for ML systems
- Chaos engineering for ML
- Disaster recovery planning
- Rollback strategies

### Week 20: Performance Optimization
- Model optimization techniques
- Quantization and pruning
- Hardware acceleration (GPU/TPU)
- Caching strategies

## Module 6: Advanced Topics (Weeks 21-24)

### Week 21: Distributed Training
- Data parallelism
- Model parallelism
- Horovod and distributed frameworks
- Cloud training platforms

### Week 22: AutoML & Hyperparameter Tuning
- Hyperparameter optimization
- AutoML platforms
- Neural Architecture Search
- Optuna/Ray Tune

### Week 23: LLMs in Production
- LLM deployment challenges
- Prompt engineering
- Fine-tuning strategies
- Vector databases
- RAG architectures

### Week 24: Security & Privacy
- Model security best practices
- Adversarial attacks and defenses
- Differential privacy
- Federated learning basics
- Compliance (GDPR, CCPA)

## Capstone Project (Weeks 25-26)

Build an end-to-end ML system incorporating:
- Data pipeline
- Model training pipeline
- CI/CD integration
- Deployment to production
- Monitoring and alerting
- Documentation

## Recommended Resources

### Books
- "Designing Machine Learning Systems" by Chip Huyen
- "Machine Learning Engineering" by Andriy Burkov
- "Building Machine Learning Powered Applications" by Emmanuel Ameisen
- "Practical MLOps" by Noah Gift & Alfredo Deza

### Online Courses
- Fast.ai Practical Deep Learning
- Andrew Ng's Machine Learning Course
- Google Cloud ML Engineering Path
- AWS ML Specialty Certification

### Tools to Master
- **Version Control**: Git, DVC
- **Orchestration**: Airflow, Kubeflow
- **Monitoring**: Prometheus, Grafana, Evidently
- **Deployment**: Docker, Kubernetes, Helm
- **Cloud**: AWS SageMaker, GCP Vertex AI, Azure ML
- **Frameworks**: TensorFlow, PyTorch, Scikit-learn

### Hands-on Labs
- Set up a complete MLOps pipeline
- Deploy a model with canary releases
- Implement feature store
- Build a model monitoring dashboard
- Create a data validation pipeline