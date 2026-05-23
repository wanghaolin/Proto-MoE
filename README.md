# Prototype-guided mixture-of-experts enables subgroup-specific risk modeling for pediatric drug-induced liver injury

## Project Overview

This project implements a Prototype-Guided Mixture-of-Experts (Proto-MoE) model for predictive modeling of pediatric drug-induced liver injury.

## Project Structure

```
Proto-MoE-github/
├── baseline.py              # Traditional ML and tabular deep learning baseline models
├── Graph-Construction/      # Graph construction module
│   ├── Biomedical_Knowledge_Graph.py   # Heterogeneous biomedical knowledge graph construction
│   └── Patient_Similarity_Graph.py     # Homogeneous patient similarity graph construction
├── homogeneous/             # Homogeneous graph models
│   ├── GCN.py               # GCN-based Proto-MoE
│   ├── GAT.py               # GAT-based Proto-MoE
│   ├── GIN.py               # GIN-based Proto-MoE
│   └── GraphSAGE.py         # GraphSAGE-based Proto-MoE
└── Heterogeneous/           # Heterogeneous graph models
    ├── HAN.py               # HAN-based Proto-MoE
    ├── HetGNN.py            # HetGNN-based Proto-MoE
    ├── MAGNN.py             # MAGNN-based Proto-MoE
    └── RGCN.py              # RGCN-based Proto-MoE
```

## Core Technical Architecture

### 1. Graph Construction Module

#### Homogeneous Graph (Patient Similarity Graph)
- Constructs patient connections based on Jaccard similarity
- Computes patient similarity using drug features
- Supports k-nearest neighbor and similarity threshold filtering

#### Heterogeneous Graph (Biomedical Knowledge Graph)
- Contains 6 node types: patient, drug, symptom, disease, gene, phenotype
- Supports multiple edge types: takes, has, interacts_with, associated_with, etc.
- Integrates drug-gene interaction and gene-phenotype association data

### 2. Mixture of Experts (MoE)

The project implements the following expert types:

| Expert Type | Description |
|-------------|-------------|
| `LinearBaselineExpert` | Linear baseline model |
| `DeepResidualExpert` | Deep residual network |
| `StandardGCNExpert` | GCN graph neural network |
| `StandardGATExpert` | GAT graph attention network |
| `FullTabTransformerExpert` | Tabular Transformer |
| `FTTransformerExpert` | FT-Transformer |
| `SpecificMetaPathExpert` | Meta-path specific expert (heterogeneous) |

### 3. Prototype Layer
- Initializes prototype vectors using K-means
- Computes similarity between samples and prototypes
- Supports prototype scattering loss
- Tracks prototype usage

### 4. Routing & Aggregation Mechanism
- **FullWeightRouter**: Dynamic routing based on prototype similarity and embedding features
- **FullWeightedAggregator**: Attention-based expert output aggregation

### 5. Loss Functions
- Label Smoothing Loss
- Supervised Contrastive Loss
- Prototype Scattering Loss
- Expert Diversity Loss
- Uncertainty Weighted Loss

## Three-Stage Training Strategy

1. **Stage 1: Representation Learning**
   - Freeze router and expert layers
   - Train encoder and prototype layer
   - Optimize contrastive loss and prototype scattering loss

2. **Stage 2: Expert Training**
   - Freeze encoder and prototype layer
   - Train router and expert layers
   - Apply graph augmentation (edge dropout, feature noise)

3. **Stage 3: Fine-tuning**
   - Unfreeze all modules
   - Jointly optimize all losses

## Usage

### 1. Data Preparation

Ensure the following data files exist in the project root directory:
- `ADR-Train.csv` - Training dataset
- `ADR-Valid.csv` - Validation dataset
- `CTD.csv` - Drug-gene interaction data
- `CTD/phenotype/CTD-phenotype.csv` - Phenotype-gene association data

### 2. Build Graph Data

```python
# Build homogeneous graph
python Graph-Construction/Patient_Similarity_Graph.py

# Build heterogeneous graph
python Graph-Construction/Biomedical_Knowledge_Graph.py
```

### 3. Run Baseline Models

```python
python baseline.py
```

### 4. Run Graph Models

```python
# Homogeneous graph models
python homogeneous/GCN.py
python homogeneous/GAT.py
python homogeneous/GIN.py
python homogeneous/GraphSAGE.py

# Heterogeneous graph models
python Heterogeneous/HAN.py
python Heterogeneous/HetGNN.py
python Heterogeneous/MAGNN.py
python Heterogeneous/RGCN.py
```

## Evaluation Metrics

The project supports the following evaluation metrics:
- F1 Score
- Matthews Correlation Coefficient (MCC)
- Precision
- Recall
- AUC-ROC
- AUPRC (Area Under Precision-Recall Curve)
- G-Mean