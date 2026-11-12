# Proto-MoE: Prototype-Guided Mixture-of-Experts for Predictive Modeling of Pediatric Drug-Induced Liver Injury

A Prototype-Guided Mixture-of-Experts framework for predictive modeling of pediatric drug-induced liver injury (DILI) from electronic health record data. Proto-MoE routes each patient through prototype-conditioned expert networks built on homogeneous patient-similarity graphs or heterogeneous biomedical knowledge graphs.

## Repository Structure

```
Proto-MoE/
├── baseline.py                            # Traditional ML and tabular deep learning baselines
├── dynamic_selection.py                   # Dynamic Selection / Ensemble Selection (DS/DES) validation suite
├── requirements.txt                       # Python dependencies
├── Graph-Construction/                    # Graph construction modules
│   ├── Biomedical_Knowledge_Graph.py      # Heterogeneous biomedical knowledge graph construction
│   └── Patient_Similarity_Graph.py        # Homogeneous patient similarity graph construction
├── homogeneous/                           # Homogeneous graph Proto-MoE variants
│   ├── GCN.py                             # GCN-based Proto-MoE
│   ├── GAT.py                             # GAT-based Proto-MoE
│   ├── GIN.py                             # GIN-based Proto-MoE
│   └── GraphSAGE.py                       # GraphSAGE-based Proto-MoE
└── Heterogeneous/                         # Heterogeneous graph Proto-MoE variants
    ├── HAN.py                             # HAN-based Proto-MoE
    ├── HetGNN.py                          # HetGNN-based Proto-MoE
    ├── MAGNN.py                           # MAGNN-based Proto-MoE
    └── RGCN.py                            # RGCN-based Proto-MoE
```

## Installation

Requires Python 3.10+ and a CUDA-capable GPU for graph model training (CPU fallback is supported but slow).

```bash
pip install -r requirements.txt
```

## Data Preparation

Place the following files in the project root (they are **not** redistributed with this repository):

| File | Description |
|------|-------------|
| `ADR-Train.csv` | Internal training cohort |
| `ADR-Valid.csv` | External validation cohort (same schema) |
| `CTD.csv` | Drug–gene interaction matrix |
| `CTD/phenotype/CTD-phenotype.csv` | Phenotype–gene association matrix |

Each cohort file contains binary drug/clinical feature columns, a patient identifier column, and a binary `label` column (`1` = DILI case).

## Usage

Run all commands from the project root.

### 1. Build graph data

```bash
# Homogeneous patient similarity graph -> train_graph-homo.pt, valid_graph-homo.pt
python Graph-Construction/Patient_Similarity_Graph.py

# Heterogeneous biomedical knowledge graph -> train_graph-Hetero.pt, valid_graph-Hetero.pt
python Graph-Construction/Biomedical_Knowledge_Graph.py
```

### 2. Run baseline models

```bash
python baseline.py
```

Performs 5-fold stratified cross-validation on the training cohort to tune probability thresholds, then evaluates on the external validation cohort. Results (MCC, Precision, Recall, F1, ROC-AUC, AUPRC, G-Mean) are printed as tab-separated rows, ready to paste into a spreadsheet.

### 3. Run Proto-MoE graph models

```bash
# Homogeneous variants
python homogeneous/GCN.py
python homogeneous/GAT.py
python homogeneous/GIN.py
python homogeneous/GraphSAGE.py

# Heterogeneous variants
python Heterogeneous/HAN.py
python Heterogeneous/HetGNN.py
python Heterogeneous/MAGNN.py
python Heterogeneous/RGCN.py
```

Each script runs prototype-count selection (knee/elbow method), 5-fold cross-validation, three-stage training of the final model, independent external validation, and visualization. 

### 4. DS/DES dynamic selection suite 

```bash
python dynamic_selection.py --train path/to/train.csv --external path/to/valid.csv
```

Trains the eleven-classifier baseline pool and evaluates all 14 Dynamic Selection / Dynamic Ensemble Selection methods (deslib) under the same internal cross-validation and frozen-threshold external protocol as the graph models. 

## Core Technical Architecture

### 1. Graph Construction Module

#### Homogeneous Graph (Patient Similarity Graph)
- Constructs patient connections based on Jaccard similarity
- Computes patient similarity using binary drug-exposure features
- Supports k-nearest-neighbor and similarity-threshold filtering, plus small-component bridging

#### Heterogeneous Graph (Biomedical Knowledge Graph)
- Contains 6 node types: patient, drug, symptom, disease, gene, phenotype
- Supports multiple edge types: takes/taken_by, has/had_by, interacts_with/interacted_by, associated_with/associated_by
- Integrates drug–gene interaction and gene–phenotype association data from CTD

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

### 6. Three-Stage Training Strategy

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

