import copy
import math
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, matthews_corrcoef, \
    f1_score, confusion_matrix, average_precision_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GINConv, Linear, GCNConv
from torch_geometric.utils import dropout_edge
import numpy as np
import os
import umap
import matplotlib
import torch.utils.checkpoint as checkpoint

matplotlib.use('Agg')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================================
# [Data Augmentation]
# ==========================================
class GraphAugmentation:
    
    @staticmethod
    def edge_dropout(edge_index, drop_rate=0.1):
        if drop_rate <= 0: return edge_index
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
        return edge_index[:, mask]
    
    @staticmethod
    def feature_noise(x, noise_scale=0.1):
        if noise_scale <= 0: return x
        return x + torch.randn_like(x) * noise_scale
    
    @staticmethod
    def feature_masking(x, mask_rate=0.1):
        if mask_rate <= 0: return x
        mask = torch.rand(x.size(), device=x.device) > mask_rate
        return x * mask.float()

    @staticmethod
    def node_dropping(x, edge_index, drop_rate=0.1):
        if drop_rate <= 0: return x, edge_index
        num_nodes = x.size(0)
        keep_mask = torch.rand(num_nodes, device=x.device) > drop_rate
        x_aug = x * keep_mask.unsqueeze(1).float()
        row, col = edge_index
        edge_mask = keep_mask[row] & keep_mask[col]
        edge_index_aug = edge_index[:, edge_mask]
        
        return x_aug, edge_index_aug
    
    @staticmethod
    def node_mixup(x, y, alpha=0.2):
        if alpha <= 0: return x, y, y, 0
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        return mixed_x, y, y[index], lam
# ==========================================
# Early Stopping & Metrics
# ==========================================
class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0.001, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score, model):
        if np.isnan(score):
            return 

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        if self.verbose:
            self.trace_func(f'Validation score improved ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


def calculate_metrics(y_true, probs, return_threshold=False):

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()

    if np.isnan(probs).any():
        probs = np.nan_to_num(probs, nan=0.0)

    thresholds = np.linspace(0.01, 0.99, 500)
    best_f1 = -1
    best_threshold = 0.5

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        if np.sum(preds) == 0:
            continue
        current_f1 = f1_score(y_true, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh
    y_pred = (probs >= best_threshold).astype(int)

    metrics = {
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5,
        'AUPRC': average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0,
        'G-Mean': np.sqrt(
            precision_score(y_true, y_pred, zero_division=0) *
            recall_score(y_true, y_pred, zero_division=0)
        ),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred),
        'Threshold': best_threshold
    }

    return metrics if not return_threshold else (metrics, best_threshold)


# ==========================================
# [Visualization]
# ==========================================
def plot_comparison_distribution(raw_features, learned_embeddings, labels, proto_ids, save_path, method='t-SNE'):

    if raw_features.shape[0] > 10000:
        idx = np.random.choice(raw_features.shape[0], 10000, replace=False)
        raw_subset = raw_features[idx]
        learned_subset = learned_embeddings[idx]
        labels_subset = labels[idx]
        proto_subset = proto_ids[idx]
    else:
        raw_subset = raw_features
        learned_subset = learned_embeddings
        labels_subset = labels
        proto_subset = proto_ids

    if method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    else:  # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

    raw_2d = reducer.fit_transform(raw_subset)
    if method == 't-SNE':
        reducer_learned = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    else:
        reducer_learned = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    learned_2d = reducer_learned.fit_transform(learned_subset)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    unique_protos = np.unique(proto_subset)
    cmap = plt.get_cmap('tab20' if len(unique_protos) > 10 else 'tab10')
    
    def get_color(i):
        return cmap(i % cmap.N)

    # --- Subplot 1: Raw + Truth ---
    ax = axes[0, 0]
    ax.scatter(raw_2d[labels_subset == 0, 0], raw_2d[labels_subset == 0, 1], c='blue', alpha=0.3, s=5, label='Negative')
    ax.scatter(raw_2d[labels_subset == 1, 0], raw_2d[labels_subset == 1, 1], c='red', alpha=0.6, s=10, label='Positive')
    ax.set_title(f"{method}: Raw Features (Truth Labels)")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- Subplot 2: Raw + Proto IDs ---
    ax = axes[0, 1]
    for i, pid in enumerate(unique_protos):
        mask = proto_subset == pid
        color = get_color(i)
        ax.scatter(raw_2d[mask, 0], raw_2d[mask, 1], color=color, alpha=0.6, s=5, label=f'Proto {pid}')
    
    ax.set_title(f"{method}: Raw Features (Proto IDs)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Prototype ID")
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- Subplot 3: Learned + Truth ---
    ax = axes[1, 0]
    ax.scatter(learned_2d[labels_subset == 0, 0], learned_2d[labels_subset == 0, 1], c='blue', alpha=0.3, s=5, label='Negative')
    ax.scatter(learned_2d[labels_subset == 1, 0], learned_2d[labels_subset == 1, 1], c='red', alpha=0.6, s=10, label='Positive')
    ax.set_title(f"{method}: Learned Embeddings (Truth Labels)")
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)

    # --- Subplot 4: Learned + Proto IDs ---
    ax = axes[1, 1]
    for i, pid in enumerate(unique_protos):
        mask = proto_subset == pid
        points = learned_2d[mask]
        color = get_color(i)
        ax.scatter(points[:, 0], points[:, 1], color=color, alpha=0.6, s=10, label=f'Proto {pid}')
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                poly = MplPolygon(hull_points, 
                                  closed=True, 
                                  facecolor=color, 
                                  alpha=0.2,
                                  edgecolor=color,
                                  linewidth=2)
                ax.add_patch(poly)
            except Exception as e:
                print(f"Warning: Could not draw hull for Proto {pid}: {e}")
    ax.set_title(f"{method}: Learned Embeddings with Cluster Boundaries")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Prototype ID")
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
# ==========================================
# [Component Definitions]
# ==========================================
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.feature_weights = nn.Parameter(torch.randn(num_features, embedding_dim) / math.sqrt(embedding_dim))
        self.feature_bias = nn.Parameter(torch.zeros(num_features, embedding_dim))

    def forward(self, x):
        x_expanded = x.unsqueeze(-1)
        w_expanded = self.feature_weights.unsqueeze(0)
        b_expanded = self.feature_bias.unsqueeze(0)
        return x_expanded * w_expanded + b_expanded

class FullTabTransformerExpert(nn.Module):
    def __init__(self, num_features, output_dim=2, dim=64, depth=2, heads=4, dropout=0.3):
        super().__init__()
        self.expert_type = "FullTabTransformer"
        self.expert_confidence = 1.0
        self.tokenizer = FeatureTokenizer(num_features, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim // 2, output_dim)
        )

    def _forward_batch(self, x):
        tokens = self.tokenizer(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, tokens), dim=1)
        x_out = self.transformer(x_seq)
        cls_out = x_out[:, 0, :]
        return self.mlp_head(cls_out)

    def forward(self, x):
        BATCH_SIZE = 64
        if x.size(0) > BATCH_SIZE:
            outs = []
            for i in range(0, x.size(0), BATCH_SIZE):
                chunk = x[i:i+BATCH_SIZE]
                outs.append(self._forward_batch(chunk))
            return torch.cat(outs, dim=0)
        return self._forward_batch(x)

    def update_confidence(self, acc):
        self.expert_confidence = 0.9 * self.expert_confidence + 0.1 * acc

class FTTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(ff_dropout)
        )
    
    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

class FTTransformerExpert(nn.Module):

    def __init__(self, num_features, output_dim=2, dim=64, depth=3, heads=8, 
                 dim_head=16, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        self.expert_type = "FT-Transformer"
        self.expert_confidence = 1.0
        self.num_features = num_features
        self.dim = dim

        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            ) for _ in range(num_features)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer_layers = nn.ModuleList()
        for _ in range(depth):
            self.transformer_layers.append(
                FTTransformerBlock(dim, heads, dim_head, attn_dropout, ff_dropout)
            )
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 2, output_dim)
        )
    
    def _forward_batch(self, x):
        batch_size = x.size(0)
        feature_tokens = []
        for i, emb in enumerate(self.feature_embeddings):
            feat = x[:, i:i+1]  # [batch, 1]
            feature_tokens.append(emb(feat))  # [batch, dim]
        tokens = torch.stack(feature_tokens, dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [batch, 1+num_features, dim]
        for layer in self.transformer_layers:
            if self.training and tokens.requires_grad:
                tokens = checkpoint.checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)
        cls_output = self.norm(tokens[:, 0])
        return self.mlp_head(cls_output)
    
    def forward(self, x):
        BATCH_SIZE = 64
        if x.size(0) > BATCH_SIZE:
            outs = []
            for i in range(0, x.size(0), BATCH_SIZE):
                chunk = x[i:i+BATCH_SIZE]
                outs.append(self._forward_batch(chunk))
            return torch.cat(outs, dim=0)
        return self._forward_batch(x)
    
    def update_confidence(self, acc):
        self.expert_confidence = 0.9 * self.expert_confidence + 0.1 * acc

class ImprovedPrototypeLayer(nn.Module):
    def __init__(self, embedding_dim, num_prototypes, diversity_lambda=0.5, scattering_lambda=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_prototypes = num_prototypes
        self.diversity_lambda = diversity_lambda
        self.scattering_lambda = scattering_lambda
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim) * 0.05)
        self.register_buffer('prototype_usage_ema', torch.ones(num_prototypes) / num_prototypes)
        self.usage_decay = 0.95
        self.sample_prototype_assignments = None
        self.prototype_samples = {}

    def initialize_with_kmeans(self, embeddings, mask, labels=None):
        train_embeddings = embeddings[mask].detach().cpu().numpy()
        if train_embeddings.shape[0] < self.num_prototypes: return
        cluster_centers = KMeans(n_clusters=self.num_prototypes, random_state=42, n_init=10).fit(
            train_embeddings).cluster_centers_
        with torch.no_grad():
            self.prototypes.data = torch.from_numpy(cluster_centers).float().to(embeddings.device)

    def get_prototype_scattering_loss(self):
        if self.num_prototypes <= 1: return torch.tensor(0.0, device=self.prototypes.device)
        prototypes_expanded = self.prototypes.unsqueeze(0)
        distances = torch.cdist(prototypes_expanded, prototypes_expanded, p=2).squeeze(0)
        mask = torch.eye(self.num_prototypes, device=self.prototypes.device).bool()
        distances = distances.masked_fill(mask, float('inf'))
        min_distances, _ = torch.min(distances, dim=1)
        loss = F.relu(3.0 - min_distances)
        return torch.mean(loss) * self.scattering_lambda

    def forward(self, embeddings):
        batch_size = embeddings.size(0)
        prototypes_expanded = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        embeddings_expanded = embeddings.unsqueeze(1).expand(-1, self.num_prototypes, -1)
        distances = torch.sqrt(torch.sum((embeddings_expanded - prototypes_expanded) ** 2, dim=2) + 1e-8)
        similarities = 1.0 / (1.0 + distances)
        assignment_probs = F.softmax(similarities * 5.0, dim=1)
        _, closest_prototypes = torch.max(similarities, dim=1)
        self.sample_prototype_assignments = closest_prototypes
        self.prototype_samples = {}
        for proto_idx in range(self.num_prototypes):
            self.prototype_samples[proto_idx] = (closest_prototypes == proto_idx).nonzero(as_tuple=True)[
                0].cpu().numpy()

        with torch.no_grad():
            soft_usage = torch.mean(assignment_probs, dim=0)
            if self.prototype_usage_ema.device != soft_usage.device:
                self.prototype_usage_ema = self.prototype_usage_ema.to(soft_usage.device)
            updated_ema = self.usage_decay * self.prototype_usage_ema + (1 - self.usage_decay) * soft_usage
            self.prototype_usage_ema.copy_(updated_ema)

        return distances, similarities, assignment_probs

    def get_prototype_assignments(self):
        return self.sample_prototype_assignments, self.prototype_samples

    def print_prototype_stats(self):
        if not hasattr(self, 'prototype_samples') or not self.prototype_samples: return
        total = sum(len(s) for s in self.prototype_samples.values())
        if total == 0: return
        for i in range(self.num_prototypes):
            cnt = len(self.prototype_samples.get(i, []))

class TrueDeepGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_channels)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # GCN Layer
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.res_proj = nn.Linear(input_dim, hidden_channels)
        self.final_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_dropout=0.0):
        if self.training and edge_dropout > 0.0:
             edge_index, _ = dropout_edge(edge_index, p=edge_dropout, training=True)
        curr_x = F.dropout(F.relu(self.input_proj(x)), p=self.dropout, training=self.training)
        raw_x = x
        for i, conv in enumerate(self.convs):
            x_in = curr_x
            out = conv(curr_x, edge_index)
            out = self.norms[i](out)
            out = F.elu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            curr_x = x_in + out

        gnn_out = curr_x
        raw_out = self.res_proj(raw_x)
        final_embedding = self.final_proj(gnn_out + raw_out)
        return final_embedding


class DeepResidualExpert(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dims=[128, 64], expert_name="DeepResidual"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.LayerNorm(hidden_dims[0]), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.LayerNorm(hidden_dims[1]), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dims[1], output_dim)
        )
        self.expert_confidence = 1.0;
        self.expert_type = expert_name

    def forward(self, x): return self.net(x)

    def update_confidence(self, acc): self.expert_confidence = 0.9 * self.expert_confidence + 0.1 * acc


class LinearBaselineExpert(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.2), nn.Linear(input_dim, output_dim))
        self.expert_confidence = 1.0;
        self.expert_type = "LinearBaseline"
    def forward(self, x): return self.net(x)
    def update_confidence(self, acc): self.expert_confidence = 0.9 * self.expert_confidence + 0.1 * acc

class StandardGCNExpert(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=64, num_layers=2, dropout=0.3, expert_name="GCNExpert"):
        super().__init__()
        self.expert_type = expert_name
        self.expert_confidence = 1.0
        self.dropout = dropout
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv in enumerate(self.convs):
            x_in = x
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_in
        return self.classifier(x)
    def update_confidence(self, acc):
        self.expert_confidence = 0.9 * self.expert_confidence + 0.1 * acc

class FullWeightRouter(nn.Module):
    def __init__(self, input_dim, num_experts, num_prototypes, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.num_experts = num_experts
        self.prototype_to_expert_map = nn.Parameter(
            torch.randn(num_prototypes, num_experts) * 0.5
        )
        self.router_net = nn.Sequential(
            nn.Linear(input_dim + num_prototypes, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)
        self.usage_decay = 0.99
    
    def forward(self, embeddings, proto_sims):
        router_input = torch.cat([embeddings, proto_sims], dim=1)
        base_logits = self.router_net(router_input)
        proto_contrib = torch.mm(proto_sims, self.prototype_to_expert_map)
        logits = base_logits + 0.5 * proto_contrib
        if self.training:
            logits = logits + torch.randn_like(logits) * 0.1
        expert_weights = F.softmax(logits, dim=1)

        if self.training:
            updated_ema = self.usage_decay * self.expert_usage_ema + \
                          (1 - self.usage_decay) * expert_weights.mean(0).detach()
            self.expert_usage_ema.copy_(updated_ema)
        return expert_weights, logits
    def get_usage_imbalance_loss(self):
        target = torch.ones(self.num_experts, device=self.expert_usage_ema.device) / self.num_experts
        return F.kl_div(torch.log(self.expert_usage_ema + 1e-8), target, reduction='batchmean')


class FullWeightedAggregator(nn.Module):
    def __init__(self, embedding_dim, expert_output_dim, num_experts, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim
        self.input_dim = embedding_dim + num_experts * expert_output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, expert_output_dim)
        )
    
    def forward(self, z, all_expert_outputs, expert_weights):
        batch_size = z.size(0)
        stacked_outputs = torch.stack(all_expert_outputs, dim=1)
        weighted_outputs = stacked_outputs * expert_weights.unsqueeze(-1)
        flattened = weighted_outputs.view(batch_size, -1)
        agg_input = torch.cat([z, flattened], dim=1)
        return self.mlp(agg_input)

# ==========================================
# [Losses]
# ==========================================

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        if batch_size < 2: return torch.tensor(0.0, device=embeddings.device)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(embeddings.device)
        eye_mask = torch.eye(batch_size, device=embeddings.device).bool()
        mask = mask.masked_fill(eye_mask, 0)
        anchor_dot_contrast = similarity_matrix / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class EnhancedPrototypeScatteringLoss(nn.Module):
    def __init__(self, margin=2.0, alpha=1.0, beta=0.5):
        super().__init__()
        self.margin = margin;
        self.alpha = alpha;
        self.beta = beta

    def forward(self, prototypes):
        num_prototypes = prototypes.size(0)
        if num_prototypes <= 1: return torch.tensor(0.0, device=prototypes.device)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        similarity_matrix = torch.mm(prototypes_norm, prototypes_norm.t())
        mask = torch.eye(num_prototypes, device=prototypes.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, 0)
        return torch.mean(similarity_matrix ** 2) * self.alpha

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_tasks=6):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses_dict):
        total_loss = 0
        task_weights = {}
        
        for i, (name, loss) in enumerate(losses_dict.items()):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            total_loss += weighted_loss
            task_weights[name] = precision.item()
        
        return total_loss, task_weights

def expert_diversity_loss(expert_outputs, temperature=0.5):

    num_experts = len(expert_outputs)
    diversity_loss = 0
    count = 0
    
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            sim = F.cosine_similarity(
                expert_outputs[i], 
                expert_outputs[j], 
                dim=1
            ).mean()
            diversity_loss += sim
            count += 1
    
    return diversity_loss / count if count > 0 else torch.tensor(0.0, device=expert_outputs[0].device)


# ==========================================
# [Main Model]
# ==========================================
class OptimizedGCNProtoMoE(nn.Module):
    def __init__(self, input_dim, out_channels=128, num_prototypes=10, num_experts=None, top_k=None,
                 dropout=0.3, expert_output_dim=2, device='cuda'):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.expert_output_dim = expert_output_dim
        self.device = device
        embedding_dim = out_channels
        GCN_HIDDEN = 128

        self.encoder = TrueDeepGCNEncoder(
            input_dim=input_dim, hidden_channels=GCN_HIDDEN, out_channels=embedding_dim,
            num_layers=3, dropout=dropout
        )

        self.prototype_layer = ImprovedPrototypeLayer(embedding_dim, num_prototypes, diversity_lambda=0.5)

        self.experts = nn.ModuleList([
            LinearBaselineExpert(embedding_dim, expert_output_dim),
            DeepResidualExpert(embedding_dim, expert_output_dim, expert_name="DeepRes"),

            StandardGCNExpert(input_dim, expert_output_dim, hidden_dim=64, num_layers=2, expert_name="GCN_Shallow"),
            StandardGCNExpert(input_dim, expert_output_dim, hidden_dim=64, num_layers=3, expert_name="GCN_Deep"),

            FullTabTransformerExpert(num_features=input_dim, output_dim=expert_output_dim),
            FTTransformerExpert(num_features=input_dim, output_dim=expert_output_dim)
        ])
        
        self.num_experts = len(self.experts)

        self.router = FullWeightRouter(embedding_dim, self.num_experts, num_prototypes, hidden_dim=64, dropout=dropout)
        self.aggregator = FullWeightedAggregator(embedding_dim, expert_output_dim, self.num_experts)

        self.label_smooth_loss = LabelSmoothingLoss(classes=2, smoothing=0.1)
        self.contrastive_loss = SupervisedContrastiveLoss()
        self.scatter_loss = EnhancedPrototypeScatteringLoss()
        self.weighted_loss = UncertaintyWeightedLoss(num_tasks=6)

    def initialize_prototypes_with_data(self, data):
        self.eval()
        with torch.no_grad():
            embs = self.encoder(data.x, data.edge_index)
            self.prototype_layer.initialize_with_kmeans(embs, mask=data.train_mask, labels=data.y)
        self.train()

    def compute_loss(self, out, labels, mask, stage=3):
        task_loss = self.label_smooth_loss(out['final_logits'][mask], labels[mask])
        cont_loss = self.contrastive_loss(out['embeddings'][mask], labels[mask])
        proto_loss = self.scatter_loss(out['prototypes'])

        bal_loss = self.router.get_usage_imbalance_loss()
        avg_proto_usage = torch.mean(out['assignment_probs'], dim=0)
        proto_entropy = -torch.sum(avg_proto_usage * torch.log(avg_proto_usage + 1e-8))
        entropy_loss = -proto_entropy * 0.5

        valid_expert_outputs = [o[mask] for o in out['all_expert_outputs']]
        div_loss = expert_diversity_loss(valid_expert_outputs)
        
        if stage == 1:
            total = cont_loss + proto_loss
            logs = {'cont': cont_loss.item(), 'proto': proto_loss.item(), 'task': 0.0}
            return total, logs
        losses_dict = {
            'task': task_loss,
            'contrast': cont_loss,
            'scatter': proto_loss,
            'balance': bal_loss,
            'entropy': entropy_loss,
            'diversity': div_loss
        }
        
        total_loss, weights = self.weighted_loss(losses_dict)
        logs = {k: v.item() if isinstance(v, torch.Tensor) else v for k,v in losses_dict.items()}
        logs.update(weights)
        
        return total_loss, logs

    def forward(self, data, edge_dropout=0.0):
        embeddings = self.encoder(data.x, data.edge_index, edge_dropout=edge_dropout)
        raw_x = data.x

        _, proto_sims, assign_probs = self.prototype_layer(embeddings)
        expert_weights, expert_logits = self.router(embeddings, proto_sims)

        all_outs = []
        for exp in self.experts:
            if isinstance(exp, (FullTabTransformerExpert, FTTransformerExpert)):
                all_outs.append(exp(raw_x))
            elif isinstance(exp, StandardGCNExpert):
                all_outs.append(exp(raw_x, data.edge_index))
            else:
                all_outs.append(exp(embeddings))
        final_out = self.aggregator(embeddings, all_outs, expert_weights)
        
        return {
            'final_logits': final_out, 
            'embeddings': embeddings, 
            'prototypes': self.prototype_layer.prototypes, 
            'expert_logits': expert_logits, 
            'expert_weights': expert_weights,
            'prototype_similarities': proto_sims, 
            'assignment_probs': assign_probs,
            'all_expert_outputs': all_outs
        }

    def freeze_modules(self, names):
        for n, m in self.named_children():
            if n in names:
                for p in m.parameters(): p.requires_grad = False

    def unfreeze_modules(self, names):
        for n, m in self.named_children():
            if n in names:
                for p in m.parameters(): p.requires_grad = True

    def analyze_prototype_expert_mapping(self, data):
        self.eval()
        with torch.no_grad():
            _ = self(data)
            mapping = self.router.prototype_to_expert_map.detach().cpu().numpy()
            print("\n=== Prototype-to-Expert Preference Matrix ===")
            for i in range(self.num_prototypes):
                expert_weights = mapping[i]
                sorted_idx = np.argsort(expert_weights)[::-1]
                s = ", ".join([f"{self.experts[idx].expert_type}({expert_weights[idx]:.2f})" for idx in sorted_idx])
                print(f"  Proto {i} -> {s}")

    def print_detailed_analysis(self, data, export_dir=None):
        self.analyze_prototype_expert_mapping(data)
        self.prototype_layer.print_prototype_stats()

        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            self.eval()
            with torch.no_grad():
                raw_x = data.x.detach().cpu().numpy()
                out = self(data)
                learned_x = out['embeddings'].detach().cpu().numpy()
                labels = data.y.detach().cpu().numpy()
                _, proto_ids = torch.max(out['prototype_similarities'], dim=1)
                proto_ids = proto_ids.detach().cpu().numpy()

                plot_comparison_distribution(
                    raw_x, learned_x, labels, proto_ids,
                    os.path.join(export_dir, 'tsne_comparison.png'), method='t-SNE'
                )
                plot_comparison_distribution(
                    raw_x, learned_x, labels, proto_ids,
                    os.path.join(export_dir, 'umap_comparison.png'), method='UMAP'
                )
                print(f"\n[Export] Saving prototype samples to {export_dir}...")
                assignments, samples_dict = self.prototype_layer.get_prototype_assignments()

                for proto_id, indices in samples_dict.items():
                    if len(indices) > 0:
                        sample_features = raw_x[indices]
                        save_path = os.path.join(export_dir, f'proto_{proto_id}_samples.csv')
                        pd.DataFrame(sample_features).to_csv(save_path, index=False)


def evaluate_gin_proto_moe(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        if hasattr(data, 'val_mask') and data.val_mask is not None:
            mask = data.val_mask
        else:
            mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)
        probs = F.softmax(out['final_logits'][mask], dim=1)[:, 1].cpu().numpy()
        y = data.y[mask].cpu().numpy()
        return calculate_metrics(y, probs)


def evaluate_individual_experts(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(data.x, data.edge_index)
        raw_x = data.x

        if hasattr(data, 'val_mask') and data.val_mask is not None:
            mask = data.val_mask
        else:
            mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)
        y = data.y[mask].cpu().numpy()
        perfs = {}

        for i, exp in enumerate(model.experts):
            if isinstance(exp, (FullTabTransformerExpert, FTTransformerExpert)):
                logits = exp(raw_x)[mask]
            elif isinstance(exp, StandardGCNExpert):
                logits = exp(raw_x, data.edge_index)[mask]
            else:
                logits = exp(embeddings[mask])

            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            f1 = 0.0
            if len(np.unique(y)) > 1:
                f1 = f1_score(y, (probs >= 0.5).astype(int), zero_division=0)
            perfs[exp.expert_type] = {'f1': f1, 'conf': exp.expert_confidence}
            exp.update_confidence(f1)
        return perfs


def evaluate_on_validation_set(model, valid_data):
    model.eval()
    with torch.no_grad():
        try:
            out = model(valid_data)
            mask = torch.ones(valid_data.num_nodes, dtype=torch.bool, device=valid_data.x.device)
            probs = F.softmax(out['final_logits'][mask], dim=1)[:, 1].cpu().numpy()
            y = valid_data.y[mask].cpu().numpy()
            metrics, thresh = calculate_metrics(y, probs, return_threshold=True)
            expert_perf = evaluate_individual_experts(model, valid_data)
            return metrics, thresh, expert_perf
        except Exception as e:
            print(f"Valid set eval error: {e}");
            import traceback;
            traceback.print_exc()
            return {}, 0.5, {}


def enhanced_three_stage_training(model, train_data, epochs_per_stage=500, lr=0.001):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-5)
    
    scaler = torch.amp.GradScaler('cuda')
    
    y_true = train_data.y
    train_mask = train_data.train_mask

    print("\n>>> Stage 1: Representation Learning (Warmup)")
    model.initialize_prototypes_with_data(train_data)
    model.freeze_modules(['router', 'experts', 'aggregator'])
    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
             out = model(train_data)
             loss, logs = model.compute_loss(out, y_true, train_mask, stage=1)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if epoch % 50 == 0:
            print(f"  Stage 1 | Ep {epoch}: Loss {loss.item():.4f} (Cont={logs.get('cont', 0):.4f})")

    print("\n>>> Stage 2: Expert Training (With Early Stopping)")
    model.initialize_prototypes_with_data(train_data)
    model.freeze_modules(['encoder', 'prototype_layer'])
    model.unfreeze_modules(['router', 'experts', 'aggregator'])
    early_stopping = EarlyStopping(patience=30, verbose=False, path='stage2_model.pt')
    
    for epoch in range(1, epochs_per_stage + 1):
        model.train()
        optimizer.zero_grad()
        
        aug_edge_index = GraphAugmentation.edge_dropout(train_data.edge_index, drop_rate=0.2)
        aug_x = GraphAugmentation.feature_noise(train_data.x, noise_scale=0.1)
        aug_data = train_data.clone()
        aug_data.edge_index = aug_edge_index
        aug_data.x = aug_x
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
             out = model(aug_data)
             loss, logs = model.compute_loss(out, y_true, train_mask, stage=2)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if epoch % 10 == 0:
            val_metrics = evaluate_gin_proto_moe(model, train_data)
            early_stopping(val_metrics['F1'], model)
            if early_stopping.early_stop:
                print("  Early stopping triggered in Stage 2")
                break
        if epoch % 50 == 0:
            print(f"  Stage 2 | Ep {epoch}: Loss {loss.item():.4f} (Task={logs.get('task', 0):.4f})")

    if os.path.exists('stage2_model.pt'):
        model.load_state_dict(torch.load('stage2_model.pt', weights_only=False))

    print("\n>>> Stage 3: Co-Evolution (Full Training)")
    model.unfreeze_modules(['encoder', 'prototype_layer', 'router', 'experts', 'aggregator'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=30, verbose=True, path='final_best_model.pt')

    for epoch in range(1, epochs_per_stage + 1):
        model.train()
        optimizer.zero_grad()
        
        aug_x1 = GraphAugmentation.feature_masking(train_data.x, mask_rate=0.15)
        aug_edge1 = GraphAugmentation.edge_dropout(train_data.edge_index, drop_rate=0.2)
        aug_data1 = train_data.clone()
        aug_data1.x = aug_x1
        aug_data1.edge_index = aug_edge1
        
        aug_x2, aug_edge2 = GraphAugmentation.node_dropping(train_data.x, train_data.edge_index, drop_rate=0.1)
        aug_x2 = GraphAugmentation.feature_noise(aug_x2, noise_scale=0.1)
        aug_data2 = train_data.clone()
        aug_data2.x = aug_x2
        aug_data2.edge_index = aug_edge2
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
             out1 = model(aug_data1)
             loss1, _ = model.compute_loss(out1, y_true, train_mask, stage=3)
             
             out2 = model(aug_data2)
             loss2, _ = model.compute_loss(out2, y_true, train_mask, stage=3)
             
             p = F.softmax(out1['final_logits'][train_mask], dim=1)
             q = F.softmax(out2['final_logits'][train_mask], dim=1)
             kl_loss = 0.5 * (F.kl_div(p.log(), q, reduction='batchmean') + F.kl_div(q.log(), p, reduction='batchmean'))
             total_loss = 0.5 * (loss1 + loss2) + 2.0 * kl_loss
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if epoch % 10 == 0:
            val_metrics = evaluate_gin_proto_moe(model, train_data)
            early_stopping(val_metrics['F1'], model)
            if early_stopping.early_stop:
                print("  Early stopping triggered in Stage 3")
                break
            print(f"  Stage 3 | Ep {epoch}: Loss {total_loss.item():.4f}, Val F1 {val_metrics['F1']:.4f}")

    if os.path.exists('final_best_model.pt'):
        model.load_state_dict(torch.load('final_best_model.pt', weights_only=False))
        print(f"✓ Restored Final Best Model")

    for f in ['stage2_model.pt', 'final_best_model.pt', 'best_checkpoint.pt', 'checkpoint.pt']:
        if os.path.exists(f): os.remove(f)
        
    return evaluate_gin_proto_moe(model, train_data)


def train_gcn_proto_moe_cv(train_data, save_dir='model', n_folds=5, epochs_per_stage=500, dynamic_k=None):
    num_nodes = train_data.num_nodes
    y_np = train_data.y.cpu().numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    device = train_data.x.device
    results = []
    all_metrics = {'F1': [], 'MCC': [], 'AUC-ROC': [], 'Precision': [], 'Recall': [], 'AUPRC': [], 'G-Mean': []}
    model_paths = []

    final_k = dynamic_k if dynamic_k is not None else 10

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(num_nodes), y_np)):
        print(f"\n=== Fold {fold + 1} ===")
        data_fold = train_data.clone()
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask[val_idx] = True
        data_fold.train_mask = train_mask
        data_fold.val_mask = val_mask

        model = OptimizedGCNProtoMoE(
            input_dim=train_data.x.size(1), out_channels=128,
            num_prototypes=final_k, device=device
        ).to(device)

        metrics = enhanced_three_stage_training(model, data_fold, epochs_per_stage=epochs_per_stage)
        print(f"Fold {fold + 1} Metrics: {metrics}")
        
        results.append({'fold': fold + 1, 'metrics': metrics})
        for k in all_metrics.keys():
            if k in metrics: all_metrics[k].append(metrics[k])
            
        fold_path = os.path.join(save_dir, f'fold{fold + 1}.pth')
        torch.save(model.state_dict(), fold_path)
        model_paths.append(fold_path)
        fold_export_dir = os.path.join(os.path.dirname(save_dir), 'prototype_samples', f'fold{fold + 1}_analysis')
        os.makedirs(fold_export_dir, exist_ok=True)
        model.print_detailed_analysis(data_fold, fold_export_dir)
    print(f"\n{'=' * 30} CV Results Summary {'=' * 30}")
    for k, v in all_metrics.items():
        mean, std = np.mean(v), np.std(v)
        print(f"{k}: {mean:.4f} ± {std:.4f}")
    return {'fold_results': results, 'summary': {k: (np.mean(v), np.std(v)) for k, v in all_metrics.items()}, 'model_paths': model_paths}


# ==========================================
# [Elbow Method]
# ==========================================
def find_optimal_prototypes_elbow(data, input_dim, device, output_dir, k_range=(2, 21)):

    temp_encoder = TrueDeepGCNEncoder(
        input_dim=input_dim, hidden_channels=128, out_channels=128,
        num_layers=3, dropout=0.0
    ).to(device)
    temp_encoder.eval()

    with torch.no_grad():
        embeddings = temp_encoder(data.x, data.edge_index).cpu().numpy()

    sse = []
    K_list = list(range(k_range[0], k_range[1]))

    for k in K_list:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)

    p1 = np.array([K_list[0], sse[0]])
    p2 = np.array([K_list[-1], sse[-1]])

    max_dist = 0
    best_k = K_list[0]

    for i, k in enumerate(K_list):
        p0 = np.array([k, sse[i]])
        dist = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
        if dist > max_dist:
            max_dist = dist
            best_k = k

    plt.figure(figsize=(10, 6))
    plt.plot(K_list, sse, 'bx-')
    plt.plot(best_k, sse[K_list.index(best_k)], 'ro', markersize=12, label=f'Elbow (K={best_k})')
    plt.xlabel('Number of Prototypes (k)')
    plt.ylabel('Inertia (SSE)')
    plt.title('Elbow Method For Optimal k')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, 'elbow_curve.png')
    plt.savefig(save_path)
    plt.close()
    return best_k


def main_gcn_proto_moe():
    CONFIG = {
        'train_graph_path': 'train_graph-homo.pt',
        'valid_graph_path': 'valid_graph-homo.pt',
        'output_dir': 'F:\\GCNProtoMoE',
        'model_save_dir': 'F:\\models',
        'prototype_export_dir': 'F:\\prototype_samples',
        'num_prototypes': 10,
        'num_experts': None, # Dynamic
        'use_elbow_method': True
    }
    for d in CONFIG.values():
        if isinstance(d, str) and not d.endswith('.pt'): os.makedirs(d, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    try:
        print("Loading Graph Data...")
        train_data = torch.load(CONFIG['train_graph_path'], weights_only=False).to(device)
        print(f"Train Data Info: {train_data}")

        input_dim = train_data.x.size(1)

        temp_experts_count = 6
        
        if CONFIG['use_elbow_method']:
            optimal_k = find_optimal_prototypes_elbow(
                train_data, input_dim, device, CONFIG['output_dir']
            )
            if optimal_k < temp_experts_count:
                optimal_k = temp_experts_count

            CONFIG['num_prototypes'] = optimal_k
            print(f"=== num_prototypes = {CONFIG['num_prototypes']} ===")

        cv_res = train_gcn_proto_moe_cv(
            train_data, CONFIG['model_save_dir'], n_folds=5, epochs_per_stage=500,
            dynamic_k=CONFIG['num_prototypes']
        )
        torch.save(cv_res, os.path.join(CONFIG['output_dir'], 'cv_results.pkl'))
        print("\n=== Visualization for Best Fold Model ===")
        best_fold_model = OptimizedGCNProtoMoE(
             input_dim=input_dim, out_channels=128,
             num_prototypes=CONFIG['num_prototypes'], device=device
        ).to(device)
        best_fold_model.load_state_dict(torch.load(cv_res['model_paths'][0], weights_only=False)) 
        best_fold_model.print_detailed_analysis(train_data, CONFIG['prototype_export_dir'])

        print("\n=== Independent Validation (Using Best Fold Model) ===")
        valid_data = torch.load(CONFIG['valid_graph_path'], weights_only=False).to(device)
        valid_data.val_mask = torch.ones(valid_data.num_nodes, dtype=torch.bool, device=device)

        metrics, thresh, exp_perf = evaluate_on_validation_set(best_fold_model, valid_data)
        print(f"Validation Metrics: {metrics}")
        pd.DataFrame([metrics]).to_csv(os.path.join(CONFIG['output_dir'], 'validation_metrics.csv'), index=False)

    except Exception as e:
        print(f"Critical Error: {e}");
        import traceback;
        traceback.print_exc()


if __name__ == "__main__":
    main_gcn_proto_moe()