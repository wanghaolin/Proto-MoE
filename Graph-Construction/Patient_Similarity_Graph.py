import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from networkx import connected_components
import networkx as nx

def load_and_preprocess_data(train_path, valid_path):
    train_df = pd.read_csv(train_path, encoding='gb2312', low_memory=False)
    valid_df = pd.read_csv(valid_path, encoding='gb2312', low_memory=False)

    patient_id_col = 'patient_id'
    label_col = 'label'

    feature_cols = [col for col in train_df.columns
                   if col not in [patient_id_col, label_col]]
    drug_cols = feature_cols[:42]
    clinical_cols = feature_cols[42:]
    print("drug_cols:", drug_cols)
    print("clinical_cols:", clinical_cols)

    def validate_features(df, name):
        if df.isnull().values.any():
            nan_count = df.isnull().sum().sum()
            raise ValueError(f"{name} contain {nan_count} missing values")

        drug_features = df[drug_cols]
        if (drug_features < 0).values.any() or (drug_features > 1).values.any():
            invalid_drug = drug_features[(drug_features < 0) | (drug_features > 1)].stack()
            raise ValueError(f"out of range：\n{invalid_drug.head()}")

        clinical_features = df[clinical_cols]
        z_scores = (clinical_features - clinical_features.mean()) / clinical_features.std()
        if (z_scores.abs() > 5).any().any():
            outlier_count = (z_scores.abs() > 5).sum().sum()
            print(f"{name} contain {outlier_count} outlier（|Z-score|>5）")

    validate_features(train_df[drug_cols + clinical_cols], "Training cohort")
    validate_features(valid_df[drug_cols + clinical_cols], "validation cohort")

    def build_features(df):
        drug_features = torch.tensor(df[drug_cols].values, dtype=torch.float)
        clinical_features = torch.tensor(df[clinical_cols].values, dtype=torch.float)
        return torch.cat([drug_features, clinical_features], dim=1)
    return (
        train_df, valid_df,
        build_features(train_df), build_features(valid_df),
        train_df[patient_id_col].values, valid_df[patient_id_col].values,
        train_df[label_col].values, valid_df[label_col].values,
        drug_cols
    )

class GraphBuilder:
    def __init__(self, k_neighbors=50, sim_threshold=0.3, min_components=3):
        self.k = k_neighbors
        self.sim_threshold = sim_threshold
        self.min_components = min_components

    def _calculate_jaccard(self, drug_matrix):
        if np.isnan(drug_matrix).any():
            raise ValueError("NaN")
        if (drug_matrix < 0).any() or (drug_matrix > 1).any():
            raise ValueError("out of 0-1")

        intersection = np.dot(drug_matrix, drug_matrix.T)
        row_sums = drug_matrix.sum(axis=1)
        union = row_sums[:, None] + row_sums - intersection
        return intersection / (union + 1e-8)

    def _calculate_jaccard(self, drug_matrix):
        """Jaccard"""
        intersection = np.dot(drug_matrix, drug_matrix.T)
        row_sums = drug_matrix.sum(axis=1)
        union = row_sums[:, None] + row_sums - intersection
        return intersection / (union + 1e-8)

    def _get_topk_edges(self, jaccard_sim):
        edges = []
        n = jaccard_sim.shape[0]
        for i in range(n):
            topk_indices = np.argsort(jaccard_sim[i])[-(self.k + 1):-1]
            for j in topk_indices:
                if j == i:
                    continue
                if jaccard_sim[i, j] >= self.sim_threshold:
                    edges.append((i, j, jaccard_sim[i, j]))
        return edges

    def _postprocess_components(self, edge_index, edge_weight, num_nodes):
        G = nx.Graph()
        try:
            G.add_edges_from([(edge_index[0, i].item(), edge_index[1, i].item(),
                             {'weight': edge_weight[i].item()})
                          for i in range(edge_index.shape[1])])
        except Exception as e:
            print(f"edge index error：{edge_index.shape}")
            raise e
        components = sorted(connected_components(G), key=lambda x: len(x), reverse=True)
        if len(components) < self.min_components:
            main_component = components[0]
            for comp in components[1:]:
                if len(comp) < 100:
                    bridge_node = next(iter(comp))
                    main_node = next(iter(main_component))
                    G.add_edge(main_node, bridge_node, weight=self.sim_threshold)
        edge_index = []
        edge_weight = []
        for u, v, data in G.edges(data=True):
            edge_index.extend([[u, v], [v, u]])
            edge_weight.extend([data['weight'], data['weight']])
        return torch.tensor(edge_index).T, torch.tensor(edge_weight)

    def build_graph(self, df, drug_cols):
        if not all(col in df.columns for col in drug_cols):
            missing = set(drug_cols) - set(df.columns)
            raise ValueError(f"Missing drug feature column：{missing}")
        drug_matrix = df[drug_cols].values.astype(float)
        jaccard_sim = self._calculate_jaccard(drug_matrix)
        raw_edges = self._get_topk_edges(jaccard_sim)
        if not raw_edges:
            raise RuntimeError("No edges were generated, please adjust the parameters")
        rows, cols, weights = zip(*[(u, v, w) for u, v, w in raw_edges] + [(v, u, w) for u, v, w in raw_edges])
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)
        edge_index, edge_weight = self._postprocess_components(edge_index, edge_weight, len(df))
        return edge_index, edge_weight

if __name__ == "__main__":
    CONFIG = {
        'train_path': 'ADR-Train.csv',
        'valid_path': 'ADR-Valid.csv',
        'k_neighbors': 30,
        'sim_threshold': 0.3,
        'min_components': 4
    }
    (train_df, valid_df,
     x_train, x_valid,
     train_patient_ids, valid_patient_ids,
     train_labels, valid_labels,
     drug_cols) = load_and_preprocess_data(CONFIG['train_path'], CONFIG['valid_path'])
    def check_tensor(tensor, name):
        if torch.isnan(tensor).any():
            raise ValueError(f"{name}contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name}Contains Inf values")
        print(f"{name}Verification passed：Shape={tensor.shape}，Range=[{tensor.min():.2f}, {tensor.max():.2f}]")
    check_tensor(x_train, "Training feature matrix")
    check_tensor(x_valid, "Verify feature matrix")

    builder = GraphBuilder(
        k_neighbors=CONFIG['k_neighbors'],
        sim_threshold=CONFIG['sim_threshold'],
        min_components=CONFIG['min_components']
    )

    print("Building training graph...")
    train_edge_index, train_edge_weight = builder.build_graph(train_df, drug_cols)
    train_data = Data(
        x=x_train,
        edge_index=train_edge_index,
        edge_attr=train_edge_weight,
        y=torch.tensor(train_labels, dtype=torch.long),
        patient_ids=train_patient_ids
    )

    print("Building validation graph...")
    valid_edge_index, valid_edge_weight = builder.build_graph(valid_df, drug_cols)
    valid_data = Data(
        x=x_valid,
        edge_index=valid_edge_index,
        edge_attr=valid_edge_weight,
        y=torch.tensor(valid_labels, dtype=torch.long),
        patient_ids=valid_patient_ids
    )

    def validate_graph_data(data, name):
        check_tensor(data.x, f"{name}.x")
        check_tensor(data.edge_index, f"{name}.edge_index")
        check_tensor(data.edge_attr, f"{name}.edge_attr")
        print(f"{name} Number of nodes：{data.num_nodes}")
        print(f"{name} Number of edges：{data.edge_index.shape[1] // 2}")
    validate_graph_data(train_data, "Training graph")
    validate_graph_data(valid_data, "Validation graph")
    torch.save(train_data, 'train_graph-homo.pt')
    torch.save(valid_data, 'valid_graph-homo.pt')

    def check_components(data):
        G = nx.Graph()
        G.add_edges_from([(data.edge_index[0, i].item(), data.edge_index[1, i].item())
                          for i in range(data.edge_index.shape[1])])
        return list(connected_components(G))
    print(f"Number of connected components in the training graph: {len(check_components(train_data))}")
    print(f"Number of connected components in the validation graph: {len(check_components(valid_data))}")
    def print_graph_stats(data, name):
        num_edges = data.edge_index.shape[1] // 2
        avg_degree = num_edges * 2 / data.num_nodes
        print(f"{name}Average degree: {avg_degree:.2f}")
        print(f"{name}Number of edges: {num_edges}")
    print_graph_stats(train_data, "Training cohort")
    print_graph_stats(valid_data, "Validation cohort")