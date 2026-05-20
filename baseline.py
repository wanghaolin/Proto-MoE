import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (matthews_corrcoef, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, precision_recall_curve)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import math
RANDOM_STATE = 42
N_FOLDS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === TabNet ===
class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super(Sparsemax, self).__init__()
        self.dim = dim

    def forward(self, input):
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))
        dim = 1
        number_of_logits = input.size(dim)
        input = torch.clamp(input, min=-1e4, max=1e4)
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range_values = torch.arange(start=1, end=number_of_logits + 1, device=input.device).float().view(1, -1)
        bound = 1 + range_values * zs
        cumsum_zs = torch.cumsum(zs, dim=dim)
        is_gt = bound > cumsum_zs
        k = torch.max(is_gt * range_values, dim=dim, keepdim=True)[0]
        tau = (torch.sum(is_gt * zs, dim=dim, keepdim=True) - 1) / k
        tau = tau.expand_as(input)
        output = torch.max(torch.zeros_like(input), input - tau)
        return output.view(original_size)

class GBN(nn.Module):
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        if self.training:
            if x.shape[0] < self.virtual_batch_size:
                return self.bn(x)
            chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
            res = [self.bn(x_) for x_ in chunks]
            return torch.cat(res, dim=0)
        else:
            return self.bn(x)

class TabNetGLU(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super(TabNetGLU, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.bn = GBN(output_dim * 2, virtual_batch_size=virtual_batch_size, momentum=momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = x[:, :x.shape[1] // 2]
        gate = x[:, x.shape[1] // 2:]
        return out * torch.sigmoid(gate)

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers, n_glu_independent, virtual_batch_size=128, momentum=0.02):
        super(FeatureTransformer, self).__init__()
        self.shared = nn.ModuleList()
        if shared_layers:
            self.shared = shared_layers
        self.independent = nn.ModuleList()
        for _ in range(n_glu_independent):
            self.independent.append(TabNetGLU(input_dim, output_dim, virtual_batch_size, momentum))
        self.scale = torch.sqrt(torch.tensor(0.5))

    def forward(self, x):
        x = self.shared[0](x)
        for glu in self.shared[1:]:
            x = torch.add(x, glu(x)) * self.scale
        for glu in self.independent:
            x = torch.add(x, glu(x)) * self.scale
        return x

class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.sparsemax(x)
        return x

class NativeTabNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_d=16, n_a=16, n_steps=3, gamma=1.3,
                 n_independent=2, n_shared=2, virtual_batch_size=128, momentum=0.02):
        super(NativeTabNet, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma

        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(TabNetGLU(input_dim, n_d + n_a, virtual_batch_size, momentum))
        for _ in range(n_shared - 1):
            self.shared_layers.append(TabNetGLU(n_d + n_a, n_d + n_a, virtual_batch_size, momentum))

        self.feat_transformers = nn.ModuleList()
        self.att_transformers = nn.ModuleList()

        for _ in range(n_steps):
            self.feat_transformers.append(FeatureTransformer(
                n_d + n_a, n_d + n_a, self.shared_layers, n_independent, virtual_batch_size, momentum
            ))
            self.att_transformers.append(AttentiveTransformer(
                n_a, input_dim, virtual_batch_size, momentum
            ))

        self.final_mapping = nn.Linear(n_d, output_dim)
        self.bn = nn.BatchNorm1d(input_dim, momentum=0.01)

    def forward(self, x):
        x = self.bn(x)
        batch_size = x.size(0)
        priors = torch.ones(batch_size, self.input_dim).to(x.device)
        out_accum = 0
        att = self.feat_transformers[0](x)
        for step in range(self.n_steps):
            M = self.att_transformers[step](priors, att[:, self.n_d:])
            priors = priors * (self.gamma - M)
            masked_x = x * M
            att = self.feat_transformers[step](masked_x)
            out = att[:, :self.n_d]
            out = F.relu(out)
            out_accum = out_accum + out
        logits = self.final_mapping(out_accum)
        return logits

class TabNetPackageExpert(nn.Module):
    def __init__(self, input_dim, output_dim=2, n_d=16, n_a=16, n_steps=3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.tabnet = NativeTabNet(
            input_dim=input_dim, output_dim=output_dim, n_d=n_d, n_a=n_a,
            n_steps=n_steps, virtual_batch_size=128, momentum=0.02
        )

    def forward(self, x):
        x = self.input_bn(x)
        logits = self.tabnet(x)
        return logits

# === TabTransformer ===
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
    def __init__(self, input_dim, output_dim=2, dim=64, depth=2, heads=4, dropout=0.3):
        super().__init__()
        num_features = input_dim
        self.input_bn = nn.BatchNorm1d(num_features)
        
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

    def forward(self, x):
        x = self.input_bn(x)
        tokens = self.tokenizer(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, tokens), dim=1)
        x_out = self.transformer(x_seq)
        cls_out = x_out[:, 0, :]
        return self.mlp_head(cls_out)

# === DeepTables ===
class DeepTablesExpert(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dims=[256, 128, 64], cross_layers=3, dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.wide = nn.Linear(input_dim, output_dim)
        self.cross_layers = nn.ModuleList()
        for _ in range(cross_layers):
            self.cross_layers.append(nn.Linear(input_dim, input_dim))
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.deep = nn.Sequential(*layers)
        self.fusion = nn.Linear(hidden_dims[-1] + input_dim + output_dim, output_dim)
    
    def forward(self, x):
        x = self.input_bn(x)
        wide_out = self.wide(x)
        x_0 = x
        x_cross = x
        for cross_layer in self.cross_layers:
            x_cross = x_0 * cross_layer(x_cross) + x_cross
        deep_out = self.deep(x)
        concat = torch.cat([wide_out, x_cross, deep_out], dim=-1)
        return self.fusion(concat)

# === FT-Transformer===
class FTTransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(ff_dropout)
        )
    
    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x

class FTTransformerExpert(nn.Module):
    def __init__(self, input_dim, output_dim=2, dim=64, depth=3, heads=8, 
                 dim_head=16, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        num_features = input_dim
        self.dim = dim
        self.input_bn = nn.BatchNorm1d(num_features)
        
        self.feature_embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim)) 
            for _ in range(num_features)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer_layers = nn.ModuleList()
        for _ in range(depth):
            self.transformer_layers.append(FTTransformerBlock(dim, heads, dim_head, attn_dropout, ff_dropout))
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(ff_dropout),
            nn.Linear(dim * 2, output_dim)
        )
    
    def forward(self, x):
        x = self.input_bn(x)
        batch_size = x.size(0)
        feature_tokens = []
        for i, emb in enumerate(self.feature_embeddings):
            feat = x[:, i:i+1] 
            feature_tokens.append(emb(feat))
        tokens = torch.stack(feature_tokens, dim=1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        cls_output = self.norm(tokens[:, 0])
        return self.mlp_head(cls_output)

class SklearnDeepWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, input_dim, batch_size=256, epochs=50, lr=1e-3, weight_decay=1e-4, **model_params):
        self.model_class = model_class
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_params = model_params
        self.model = None
        self.device = DEVICE

    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = self.model_class(input_dim=self.input_dim, output_dim=2, **self.model_params)
        self.model = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model needs to be fitted first.")
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X[0].to(self.device)
                logits = self.model(batch_X)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs, axis=0)
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def load_data():
    train = pd.read_csv('ADR-Train.csv', encoding='gb2312', low_memory=False)
    valid = pd.read_csv('ADR-Valid.csv', encoding='gb2312',low_memory=False)

    X_train = train.drop(columns=['label'])
    y_train = train['label']
    X_valid = valid.drop(columns=['label'])
    y_valid = valid['label']
    X_train = X_train.astype(float)
    X_valid = X_valid.astype(float)

    return X_train, y_train, X_valid, y_valid

def find_optimal_threshold(y_true, y_proba):

    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_proba, np.ndarray):
        y_proba = np.array(y_proba)

    if np.isnan(y_proba).any():
        y_proba = np.nan_to_num(y_proba, nan=0.0)

    thresholds = np.linspace(0.01, 0.99, 500)
    best_f1 = -1
    best_threshold = 0.5

    for thresh in thresholds:
        preds = (y_proba >= thresh).astype(int)
        if np.sum(preds) == 0:
            continue
        current_f1 = f1_score(y_true, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh

    return best_threshold


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    X_train, y_train, X_valid, y_valid = load_data()
    input_dim = X_train.shape[1]
    print(f"Data Loaded. Input Dimensions: {input_dim}")

    MODELS = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=7, max_features='sqrt', criterion='entropy', bootstrap=False, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1, random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1),
        'LGBM': LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, min_child_samples=1, max_depth=5, verbose=-1, min_child_weight=0.01, n_jobs=-1),
        'LogisticRegression': LogisticRegression(C=0.3, random_state=RANDOM_STATE),
        'SVM': SVC(C=3, kernel='rbf', probability=True, gamma='scale', random_state=RANDOM_STATE),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE),
        'LDA': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, subsample=0.8,max_features='sqrt',n_iter_no_change=10, random_state=RANDOM_STATE),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE, algorithm='SAMME.R'),
        'QDA': QuadraticDiscriminantAnalysis(reg_param=0.3),
        'catboost': CatBoostClassifier(iterations=200, learning_rate=0.01,depth=6,l2_leaf_reg=5,random_seed=RANDOM_STATE, verbose=0,task_type='GPU')
    }
    print("Initializing Deep Learning Tabular Experts...")

    # 1. TabNet
    MODELS['TabNet'] = SklearnDeepWrapper(
        model_class=TabNetPackageExpert,
        input_dim=input_dim,
        epochs=100,
        lr=0.02,
        n_d=16, n_a=16, n_steps=3
    )
    
    # 2. TabTransformer
    MODELS['TabTransformer'] = SklearnDeepWrapper(
        model_class=FullTabTransformerExpert,
        input_dim=input_dim,
        epochs=100,
        dim=32, depth=2, heads=4
    )
    
    # 3. DeepTables
    MODELS['DeepTables'] = SklearnDeepWrapper(
        model_class=DeepTablesExpert,
        input_dim=input_dim,
        epochs=100,
        hidden_dims=[128, 64],
        cross_layers=2
    )
    
    # 4. FT-Transformer
    MODELS['FT-Transformer'] = SklearnDeepWrapper(
        model_class=FTTransformerExpert,
        input_dim=input_dim,
        epochs=100,
        dim=32, depth=3, heads=4
    )
    
    print(f"Total Models to Train: {len(MODELS)}")
    cv_results = {name: [] for name in MODELS}
    final_results = {}
    optimal_thresholds = {}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_valid_np = X_valid.values
    y_valid_np = y_valid.values

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
        X_fold_train, X_fold_val = X_train_np[train_idx], X_train_np[val_idx]
        y_fold_train, y_fold_val = y_train_np[train_idx], y_train_np[val_idx]

        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = scaler.transform(X_fold_val)

        for name, model in MODELS.items():
            print(f"Training {name}...", end=" ")
            try:
                cloned_model = clone(model)
                cloned_model.fit(X_fold_train_scaled, y_fold_train)

                y_proba = cloned_model.predict_proba(X_fold_val_scaled)[:, 1]

                optimal_threshold = find_optimal_threshold(y_fold_val, y_proba)
                y_pred = (y_proba >= optimal_threshold).astype(int)

                cv_results[name].append({
                    'mcc': matthews_corrcoef(y_fold_val, y_pred),
                    'precision': precision_score(y_fold_val, y_pred, pos_label=1, zero_division=0),
                    'recall': recall_score(y_fold_val, y_pred, pos_label=1, zero_division=0),
                    'f1': f1_score(y_fold_val, y_pred, pos_label=1, zero_division=0),
                    'roc_auc': roc_auc_score(y_fold_val, y_proba),
                    'auprc': average_precision_score(y_fold_val, y_proba),
                    'G-Mean': np.sqrt(precision_score(y_fold_val, y_pred, pos_label=1) * recall_score(y_fold_val, y_pred, pos_label=1)),
                    'threshold': optimal_threshold
                })
                print(f"Done. AUC: {cv_results[name][-1]['roc_auc']:.4f}")
            except Exception as e:
                print(f"Error training {name}: {e}")
                cv_results[name].append({'mcc':0,'precision':0,'recall':0,'f1':0,'roc_auc':0,'auprc':0,'G-Mean':0,'threshold':0.5})

    for name in MODELS:
        thresholds = [res['threshold'] for res in cv_results[name]]
        optimal_thresholds[name] = np.mean(thresholds)
        print(f"{name} averge best thresholds: {optimal_thresholds[name]:.3f}")

    print("\n=== Training Final Models ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_np)
    X_valid_scaled = scaler.transform(X_valid_np)

    for name in MODELS:
        print(f"Final training for {name}...")
        try:
            final_model = clone(MODELS[name])
            final_model.fit(X_train_scaled, y_train_np)
            y_proba_val = final_model.predict_proba(X_valid_scaled)[:, 1]
            optimal_threshold = optimal_thresholds[name]
            y_pred_val = (y_proba_val >= optimal_threshold).astype(int)

            cm = confusion_matrix(y_valid_np, y_pred_val)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            recall = recall_score(y_valid_np, y_pred_val, pos_label=1, zero_division=0)
            gmean = np.sqrt(recall * specificity) if (recall > 0 and specificity > 0) else 0

            final_results[name] = {
                'mcc': matthews_corrcoef(y_valid_np, y_pred_val),
                'precision': precision_score(y_valid_np, y_pred_val, pos_label=1, zero_division=0),
                'recall': recall,
                'f1': f1_score(y_valid_np, y_pred_val, pos_label=1, zero_division=0),
                'roc_auc': roc_auc_score(y_valid_np, y_proba_val),
                'auprc': average_precision_score(y_valid_np, y_proba_val),
                'G-Mean': gmean,
                'threshold': optimal_threshold
            }

            print(f"Confusion matrix for {name}:")
            print(cm)
            print(f"specificity: {specificity:.3f}, recall: {recall:.3f}, G-Mean: {gmean:.3f}, threshold: {optimal_threshold:.3f}\n")
        except Exception as e:
            print(f"Error in final evaluation for {name}: {e}")
            final_results[name] = {'mcc':0,'precision':0,'recall':0,'f1':0,'roc_auc':0,'auprc':0,'G-Mean':0,'threshold':0}

    print("\n=== Cross-Validation Results ===")
    print("Model\tMCC\tPrecision\tRecall\tF1\tROC_AUC\tAUPRC\tG-Mean")
    for name in MODELS:
        mcc_list = [res['mcc'] for res in cv_results[name]]
        precision_list = [res['precision'] for res in cv_results[name]]
        recall_list = [res['recall'] for res in cv_results[name]]
        f1_list = [res['f1'] for res in cv_results[name]]
        roc_auc_list = [res['roc_auc'] for res in cv_results[name]]
        auprc_list = [res['auprc'] for res in cv_results[name]]
        gmean_list = [res['G-Mean'] for res in cv_results[name]]
        def get_mean_std(data):
            return np.mean(data), np.std(data)

        avg_mcc, std_mcc = get_mean_std(mcc_list)
        avg_precision, std_precision = get_mean_std(precision_list)
        avg_recall, std_recall = get_mean_std(recall_list)
        avg_f1, std_f1 = get_mean_std(f1_list)
        avg_roc_auc, std_roc_auc = get_mean_std(roc_auc_list)
        avg_auprc, std_auprc = get_mean_std(auprc_list)
        avg_gmean, std_gmean = get_mean_std(gmean_list)

        excel_format = f"{name}\t{avg_mcc:.3f}±{std_mcc:.3f}\t{avg_precision:.3f}±{std_precision:.3f}\t{avg_recall:.3f}±{std_recall:.3f}\t{avg_f1:.3f}±{std_f1:.3f}\t{avg_roc_auc:.3f}±{std_roc_auc:.3f}\t{avg_auprc:.3f}±{std_auprc:.3f}\t{avg_gmean:.3f}±{std_gmean:.3f}"
        print(excel_format)

    print("\n=== External Validation Results ===")
    print("Dataset\tMCC\tPrecision\tRecall\tF1\tROC_AUC\tAUPRC\tG-Mean")
    for name, res in final_results.items():
        row = f"{name}\t{res['mcc']:.3f}\t{res['precision']:.3f}\t{res['recall']:.3f}\t{res['f1']:.3f}\t{res['roc_auc']:.3f}\t{res['auprc']:.3f}\t{res['G-Mean']:.3f}"
        print(row)

if __name__ == "__main__":
    main()