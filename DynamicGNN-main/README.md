📘 Incremental GNN with Cached AX and SVD-Based Updates
This repository contains an experimental framework for incremental graph learning using Graph Neural Networks (GNNs). The notebook evaluates different strategies to update a trained GNN when the underlying graph evolves over time.


🚀 Overview

Real-world graphs are dynamic. Retraining a GNN from scratch after every update is computationally expensive. This project explores efficient alternatives:

. Full retraining (baseline)
. No update (stale model)
. Cached adjacency-based incremental updates
. Subgraph-based localized training
. SVD-based selective parameter updates (proposed idea)

📊 Datasets Used

We use large-scale node classification datasets from the Open Graph Benchmark:

1. ogbn-arxiv
 - ~169K nodes, ~1.1M edges
 - Task: Paper subject classification

2. ogbn-products
 - ~2.4M nodes, ~61M edges
 - Task: Product category prediction
3. Pubmed
 - ~2.4M nodes, ~61M edges
 - Task: Paper subject classification
4. Cora
 - ~3K nodes, ~5K edges
 - Task: Product category prediction
5. Synthetic_data1
 - ~10K nodes, ~600K edges
 - Task: Node classification

🧠 Key Features

-> Custom pipeline for converting OGB datasets → PyTorch Geometric format
-> Incremental graph updates via edge batching

. Multiple training strategies:
  
  -> Retrain from scratch
  
  -> Cached AX updates
  
  -> Subgraph-based fine-tuning
  
  -> Row-selective updates using SVD

. Performance comparison using:
  
  -> Accuracy
  
  -> Loss
  
  -> Training time
 
  -> Singular Value Decomposition (SVD) drift
