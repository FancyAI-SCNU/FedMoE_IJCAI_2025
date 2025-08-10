
<div align="center">

# Fed-MoE: Heterogeneous Federated Learning with Scalable Server Mixture-of-Experts

[![Paper](https://img.shields.io/badge/Paper-Fed&minus;MoE-005DAA?style=for-the-badge&logo=acm)](https://fanchenyou.github.io/docs/ijcai25.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

> **Jingang Jiang†**, **Yanzhao Chen†**, Xiangyang Liu, Haiqi Jiang, Chenyou Fan∗  
> *South China Normal University*

</div>

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#-abstract" style="text-decoration: none; font-weight: bold;">📄 Abstract</a> •
    <a href="#-key-features" style="text-decoration: none; font-weight: bold;">🔍 Key Features</a> •
    <a href="#-datasets" style="text-decoration: none; font-weight: bold;">✨ Dataset</a>
  </p>
  <p>
    <a href="#-installation" style="text-decoration: none; font-weight: bold;">📦 Installation</a> •
    <a href="#️-usage" style="text-decoration: none; font-weight: bold;">⚙️ Usage</a> •
    <a href="#-results" style="text-decoration: none; font-weight: bold;">📊 Results</a> •
    <a href="#-technical-highlights" style="text-decoration: none; font-weight: bold;">🧠 Technical Highlights</a>
  </p>
</div>

---

## 📄 Abstract

Classical Federated Learning (FL) faces challenges when deploying large models on power-constrained clients. We propose an asymmetric FL mechanism that enables the aggregation of compact client models into a comprehensive server **Mixture-of-Experts (MoE)**, allowing for efficient fusion of the most pertinent client models to update each server expert based on the measured relevance.

To address the Non-IID data issue, we optimize the server-side MoE architecture by incorporating a **main expert** that always activates alongside a set of selectively activated **routed experts**. This configuration ensures a balance between learning general knowledge and specific data distribution.

Our **Fed-MoE framework** is model-agnostic and has demonstrated notable improvements on vision FL tasks with million-scale ResNet backbones, and language tasks with billion-scale BERT and GPT-2 backbones.

---

## 🔍 Key Features

- **Asymmetric FL**: Compact clients, large MoE server
- **Dynamic Expert** Dispatching: Relevance-based aggregation
- **Main + Routed Experts**: Generalization + Specialization
- **Gating Entropy Loss**: Encourages expert diversification
- **Supports Large Models**: ResNet, BERT, GPT-2…

---

## ✨ Datasets

| Dataset  | Task Type                   | Client Count  | Model Used |
| -------- | --------------------------- | ------------- | ---------- |
| FEMNIST  | Image Classification        | 10 / 50 / 100 | CNN        |
| CIFAR-10 | Image Classification        | 10 / 50 / 100 | ResNet     |
| SENT-140 | Sentiment Analysis (Binary) | 10 / 50 / 100 | BERT       |
| Yelp     | Review Star Rating (5-way)  | 10 / 50 / 100 | GPT-2      |

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/FancyAI-SCNU/FedMoE_IJCAI_2025.git
cd FedMoE_IJCAI_2025
```

### Step 2: Install Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch~=2.6.0 numpy~=1.26.4 pandas~=2.2.2 matplotlib~=3.9.0
```

For NLP tasks (SENT140 with BERT), you'll also need:

```bash
pip install transformers tokenizers
```

---

## ⚙️ Usage

### Training Fed-MoE on Different Datasets

#### 1. CIFAR-10 (Image Classification with ResNet-18)

Navigate to the CIFAR-10 directory and run:

```bash
cd fed_moe_cifar10
python fedmoe.py --nc 50 --lr 1e-4 --cuda 0 --spc 1000 --a 0.1 --entro 1e-3
```

**Parameters:**
- `--nc`: Number of clients (default: 50)
- `--lr`: Learning rate (default: 1e-4)  
- `--cuda`: CUDA device index (default: 0)
- `--spc`: Server samples per class (default: 1000)
- `--a`: FedAvg weight parameter (default: 0.1)
- `--entro`: Entropy loss weight (default: 1e-3)

#### 2. FEMNIST (Handwritten Character Recognition)

Navigate to the FEMNIST directory and run:

```bash
cd fed_moe_femnist
python fedmoe.py --nc 50 --upc 30 --ups 10 --lr 3e-4 --cuda 0 --a 0.1 --entro 1e-1
```

**Parameters:**
- `--nc`: Number of clients (default: 50)
- `--upc`: Users per client (default: 30)
- `--ups`: Users per server (default: 10)
- `--lr`: Learning rate (default: 3e-4)
- `--cuda`: CUDA device index (default: 0)
- `--a`: FedAvg weight parameter (default: 0.1)
- `--entro`: Entropy loss weight (default: 1e-1)

#### 3. SENT140 (Sentiment Analysis with BERT)

Navigate to the SENT140 directory and run:

```bash
cd fed_moe_sent140
python fedmoe.py --nc 50 --nme 0 --lr 5e-4 --cuda 0 --npos 500 --nneg 500 --a 0.1 --entro 1e-1
```

**Parameters:**
- `--nc`: Number of clients (default: 50)
- `--nme`: Number of main experts (default: 0)
- `--lr`: Learning rate (default: 5e-4)
- `--cuda`: CUDA device index (default: 0)
- `--npos`: Server positive samples (default: 500)
- `--nneg`: Server negative samples (default: 500)
- `--a`: FedAvg weight parameter (default: 0.1)
- `--entro`: Entropy loss weight (default: 1e-1)

#### 4. Yelp (Review Rating with GPT-2)

Navigate to the Yelp directory and run:

```bash
cd fed_moe_yelp
python fedmoe.py --nc 10 --lr 3e-4 --cuda 0 --a 0.1 --entro 1e-3
```

**Parameters:**
- `--nc`: Number of clients (default: 10)
- `--lr`: Learning rate (default: 3e-4)
- `--cuda`: CUDA device index (default: 0)
- `--a`: FedAvg weight parameter (default: 0.1)
- `--entro`: Entropy loss weight (default: 1e-3)

### Example Training Commands

Quick start with different client configurations:

```bash
# CIFAR-10 with 10 clients
cd fed_moe_cifar10/CIFAR10
python fedmoe.py --nc 10 --lr 1e-4

# FEMNIST with 100 clients  
cd fed_moe_femnist
python fedmoe.py --nc 100 --lr 3e-4

# SENT140 with 50 clients
cd fedmoe_sent140/SENT140
python fedmoe.py --nc 50 --lr 5e-4

# Yelp with 10 clients
cd fed_moe_yelp
python fedmoe.py --nc 10 --lr 3e-4
```

### Output and Monitoring

Each training run will:
- Create TensorBoard logs in `runs/` directory
- Print training progress and metrics to console
- Save model checkpoints and results
- Display final accuracy and convergence statistics

To monitor training progress with TensorBoard:

```bash
tensorboard --logdir=runs
```

---

## 📊 Results

Below are some of the results achieved using Fed-MoE across different datasets:

| Dataset  | Clients | Accuracy (%) |
| -------- | ------- | ------------ |
| FEMNIST  | 50      | 86.03        |
| FEMNIST  | 100     | 82.58        |
| CIFAR-10 | 50      | 65.52        |
| CIFAR-10 | 100     | 60.73        |
| SENT-140 | 100     | 78.10        |
| Yelp     | 100     | 53.46        |

For more ablation studies and comparisons, please refer to the [paper](https://fanchenyou.github.io/docs/ijcai25.pdf).

---

## 🧠 Technical Highlights

### Architecture Overview

Each client has a small, compact model (CNN/BERT/GPT), while the server maintains a large **Mixture-of-Experts (MoE)** composed of:

- One **main expert**
- Multiple **routed experts**

The MoE gate dynamically selects relevant experts for each input.

### Core Components

- **Stage A**: Local client training and uploading
- **Stage B**: Server MoE update with relevance-based aggregation
- **Stage C**: Client synchronization using server-client correlation matrix

