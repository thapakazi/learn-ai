---
title: "Module 1: Fundamentals"
description: "Core ML concepts and Python essentials for ML engineering"
weight: 2
sidebar:
  open: true
---

# Fundamentals Module

This module establishes the foundational knowledge required for ML Engineering, building upon your existing DevOps/SRE expertise.

## Learning Objectives

By the end of this module, you will:
- Master Python for ML engineering tasks
- Understand core mathematical concepts used in ML
- Grasp fundamental ML algorithms and their applications
- Get hands-on experience with deep learning frameworks

## Module Structure

{{< cards >}}
  {{< card link="week1" title="Week 1: Python Fundamentals" icon="code" subtitle="Python essentials with DevOps/SRE focus" >}}
  {{< card link="week2" title="Week 2: Mathematics & Statistics" icon="calculator" subtitle="Core math concepts for ML (Coming Soon)" >}}
  {{< card link="week3" title="Week 3: ML Fundamentals" icon="light-bulb" subtitle="Machine learning algorithms (Coming Soon)" >}}
  {{< card link="week4" title="Week 4: Deep Learning" icon="chip" subtitle="Neural networks basics (Coming Soon)" >}}
{{< /cards >}}

## Quick Navigation

### 📚 Week 1: Python for ML Engineering
- [Python Basics for DevOps/SRE](/docs/fundamentals/week1/python-basics/)
- [NumPy for Infrastructure Metrics](/docs/fundamentals/week1/numpy-arrays/)
- [Pandas for Log Analysis](/docs/fundamentals/week1/pandas-dataframes/)
- [Matplotlib for Monitoring Dashboards](/docs/fundamentals/week1/matplotlib-visualization/)

### 📐 Week 2: Mathematics & Statistics
*Content coming soon*
- Linear algebra fundamentals
- Statistics for ML
- Calculus essentials

### 🤖 Week 3: Machine Learning Fundamentals
*Content coming soon*
- Supervised learning algorithms
- Unsupervised learning techniques
- Model evaluation metrics

### 🧠 Week 4: Deep Learning Basics
*Content coming soon*
- Neural network architectures
- TensorFlow/PyTorch introduction
- CNNs, RNNs, and Transformers

## Assessment Checkpoints

### Week 1-2 Checkpoint
- Python coding assessment
- Mathematical problem set

### Week 3-4 Project
Build an ML pipeline that:
- Ingests operational data
- Trains a model
- Evaluates performance
- Provides predictions via API

## Resources

### Required Reading
- "Python Machine Learning" by Sebastian Raschka (Chapters 1-4)
- "The Elements of Statistical Learning" (Selected sections)

### Online Materials
- [Fast.ai Practical Deep Learning](https://course.fast.ai/) (Lessons 1-3)
- [3Blue1Brown Neural Network series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)

## Tools Setup

```bash
# Create virtual environment
python -m venv ml-env
source ml-env/bin/activate

# Install core packages
pip install numpy pandas scikit-learn
pip install tensorflow pytorch
pip install jupyter mlflow
```

## Next Steps

After completing this module, you'll be ready for [Module 2: MLOps Core](/docs/mlops), where you'll apply DevOps principles to machine learning workflows.