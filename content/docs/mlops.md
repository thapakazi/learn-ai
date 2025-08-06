---
title: Module 2 - MLOps Core
weight: 3
---

# MLOps Core Module

This module applies DevOps principles to machine learning, leveraging your existing CI/CD and automation expertise.

## Learning Objectives

- Implement version control for data and models
- Build automated ML pipelines
- Create robust CI/CD for ML systems
- Establish model governance and registry

## Week 5: Version Control for ML

### Beyond Code Versioning
- **Data Version Control (DVC)**
  ```bash
  # Initialize DVC
  dvc init
  dvc add data/training_set.csv
  git add data/training_set.csv.dvc
  git commit -m "Add training data"
  ```

- **Experiment Tracking**
  - MLflow setup and integration
  - Weights & Biases for distributed teams
  - Comparing experiment results

### Hands-on Lab
Create a versioned ML project with:
- Code in Git
- Data in DVC
- Experiments in MLflow
- Models in registry

## Week 6: ML Pipeline Orchestration

### Orchestration Platforms

#### Apache Airflow
```python
from airflow import DAG
from airflow.operators.python import PythonOperator

def train_model():
    # Training logic
    pass

with DAG('ml_training_pipeline', 
         schedule_interval='@daily') as dag:
    
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model
    )
```

#### Kubeflow Pipelines
- Component-based architecture
- Kubernetes-native execution
- Artifact tracking

### Pipeline Patterns
- Data validation gates
- Model quality checks
- Automated retraining triggers
- Multi-stage deployments

## Week 7: CI/CD for ML

### Testing Strategy

#### Unit Tests for ML
```python
def test_feature_engineering():
    data = create_test_data()
    features = engineer_features(data)
    assert features.shape[1] == expected_features
    assert not features.isnull().any()
```

#### Integration Tests
- Data pipeline validation
- Model serving endpoints
- Performance benchmarks

### Deployment Strategies
- **Blue-Green Deployments**
- **Canary Releases**
  ```yaml
  # Kubernetes canary deployment
  spec:
    replicas: 10
    strategy:
      canary:
        steps:
        - setWeight: 10
        - pause: {duration: 10m}
        - setWeight: 50
        - pause: {duration: 10m}
  ```
- **Shadow Deployments**
- **A/B Testing Framework**

## Week 8: Model Registry & Governance

### Model Registry Implementation
- Centralized model storage
- Metadata management
- Version comparison
- Promotion workflows

### MLflow Model Registry
```python
import mlflow

# Register model
mlflow.register_model(
    "runs:/run_id/model",
    "production_model"
)

# Transition stages
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="production_model",
    version=1,
    stage="Production"
)
```

### Governance Framework
- Model approval processes
- Audit trails
- Compliance documentation
- Performance SLAs

## Practical Projects

### Project 1: End-to-End Pipeline
Build a complete MLOps pipeline that:
1. Triggers on new data
2. Validates data quality
3. Trains model
4. Runs tests
5. Deploys if metrics pass
6. Monitors in production

### Project 2: A/B Testing System
Implement model A/B testing with:
- Traffic splitting
- Metric collection
- Statistical significance testing
- Automated winner selection

## Assessment Criteria

### Technical Skills
- [ ] Configure DVC for data versioning
- [ ] Build Airflow DAG for ML pipeline
- [ ] Implement CI/CD with model testing
- [ ] Deploy model with canary release

### Best Practices
- [ ] Reproducible experiments
- [ ] Automated testing coverage
- [ ] Monitoring and alerting
- [ ] Documentation

## Tools & Technologies

### Core Stack
- **Version Control**: Git, DVC
- **Experiment Tracking**: MLflow, W&B
- **Orchestration**: Airflow, Kubeflow
- **CI/CD**: GitHub Actions, GitLab CI, Jenkins
- **Registry**: MLflow, Seldon Core

### Setup Commands
```bash
# Install MLOps tools
pip install dvc mlflow airflow
pip install pytest pytest-cov
pip install seldon-core

# Configure MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Initialize Airflow
airflow db init
airflow webserver --port 8080
```

## Resources

### Documentation
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Guide](https://mlflow.org/docs/latest/index.html)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/pipelines/)

### Books & Articles
- "Introducing MLOps" by Mark Treveil
- "MLOps: Continuous delivery and automation pipelines in machine learning"
- Google's "Hidden Technical Debt in Machine Learning Systems"

## Next Module

Continue to [Module 3: Infrastructure & Deployment](/docs/infrastructure) to learn about ML-specific infrastructure patterns and scalable deployment strategies.