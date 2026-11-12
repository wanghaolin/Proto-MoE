"""Run all 14 dynamic selection (DS/DES) methods on the Proto-MoE tabular inputs."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from deslib.des.des_knn import DESKNN
from deslib.des.des_mi import DESMI
from deslib.dcs.a_posteriori import APosteriori
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.des_p import DESP
from deslib.dcs.ola import OLA
from deslib.dcs.mcb import MCB
from deslib.dcs.rank import Rank
from deslib.des.knop import KNOP
from deslib.dcs.lca import LCA
from deslib.dcs.mla import MLA
from deslib.dcs.a_priori import APriori
from deslib.des.meta_des import METADES
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# Column names of the source CSVs are Chinese because the underlying clinical
# data is Chinese. These constants keep the raw headers and document their
# English meaning wherever they appear in generated outputs.
PATIENT_ID_COLUMN = "病历号"               # "Medical Record No." (admission ID)
DISCHARGE_DEPT_CODE_COLUMN = "出院科室编码"  # "Discharge Department Code"
DISCHARGE_DEPT_COLUMN = "出院科室"          # "Discharge Department"

METRIC_COLUMNS = [
    "MCC", "F1", "Precision", "Recall", "Specificity", "AUROC", "AUPRC",
    "GMean", "GPR", "Brier",
]


def read_csv(path):
    """Preserve literal admission IDs, including leading zeroes."""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={PATIENT_ID_COLUMN: str})
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode CSV: {path}")


def load_data(train_path, external_path):
    train, external = read_csv(train_path), read_csv(external_path)
    excluded = {PATIENT_ID_COLUMN, "label", DISCHARGE_DEPT_CODE_COLUMN, DISCHARGE_DEPT_COLUMN}
    # The external file retains the original clinical-feature order. Select the
    # same named columns internally; recently appended department columns stay out.
    features = [c for c in external.columns if c not in excluded]
    if len(features) != 87 or len(set(features)) != 87:
        raise ValueError(f"Expected 87 distinct clinical features, found {len(features)}")
    matrices, labels = [], []
    for frame in (train, external):
        matrix = frame.loc[:, features].to_numpy(dtype=np.float64)
        label_values = frame["label"].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("Use the existing preprocessed inputs; missing/nonfinite features found")
        if set(np.unique(label_values)) != {0.0, 1.0}:
            raise ValueError("Both binary label classes 0 and 1 are required")
        matrices.append(matrix)
        labels.append(label_values.astype(np.int64))
    return train, external, features, matrices[0], labels[0], matrices[1], labels[1]


def make_models(seed):
    """Fresh copies of the eleven baseline classifiers used as the pool.

    Returns a list of ``(classifier, name)`` tuples, matching the iteration
    order used when fitting and when validating reused checkpoints.
    """
    models = [
        ('RandomForest', RandomForestClassifier(n_estimators=200, max_depth=7, max_features='sqrt', criterion='entropy', bootstrap=False, random_state=seed, n_jobs=-1)),
        ('XGBoost', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1, random_state=seed, eval_metric='logloss', n_jobs=-1)),
        ('LGBM', LGBMClassifier(n_estimators=200, random_state=seed, min_child_samples=1, max_depth=5, verbose=-1, min_child_weight=0.01, n_jobs=-1)),
        ('LogisticRegression', LogisticRegression(C=0.3, random_state=seed)),
        ('SVM', SVC(C=3, kernel='rbf', probability=True, gamma='scale', random_state=seed)),
        ('ExtraTrees', ExtraTreesClassifier(n_estimators=200, random_state=seed)),
        ('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.1, subsample=0.8, max_features='sqrt', n_iter_no_change=10, random_state=seed)),
        ('AdaBoost', AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=seed, algorithm='SAMME.R')),
        ('QDA', QuadraticDiscriminantAnalysis(reg_param=0.3)),
        ('catboost', CatBoostClassifier(iterations=200, learning_rate=0.01, depth=6, l2_leaf_reg=5, random_seed=seed, verbose=0)),
    ]
    return models


def pool_names():
    return [name for _, name in make_models(0)]


def metrics(y, predicted, probabilities):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = precision_score(y, predicted, zero_division=0)
    recall = recall_score(y, predicted, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "MCC": matthews_corrcoef(y, predicted),
        "F1": f1_score(y, predicted, zero_division=0), "Precision": precision,
        "Recall": recall, "Specificity": specificity,
        "AUROC": roc_auc_score(y, probabilities),
        "AUPRC": average_precision_score(y, probabilities),
        "GMean": np.sqrt(precision * recall), "GPR": np.sqrt(precision * recall),
        "Brier": brier_score_loss(y, probabilities),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

def role_rows(indices, role, fold):
    return [{"row_index": int(i), "role": role, "fold": fold} for i in indices]


def make_selectors(pool, seed, jobs):
    """Same methods and method defaults as the original DS script."""
    return [
        (DESKNN(pool, random_state=seed, n_jobs=jobs), 'DESKNN'),
        (DESMI(pool, random_state=seed, n_jobs=jobs), 'DESMI'),
        (APosteriori(pool, random_state=seed, n_jobs=jobs), 'APosteriori'),
        (KNORAU(pool, random_state=seed, n_jobs=jobs), 'KNORAU'),
        (KNORAE(pool, random_state=seed, n_jobs=jobs), 'KNORAE'),
        (DESP(pool, random_state=seed, n_jobs=jobs), 'DESP'),
        (OLA(pool, random_state=seed, n_jobs=jobs), 'OLA'),
        (MCB(pool, random_state=seed, n_jobs=jobs), 'MCB'),
        (Rank(pool, random_state=seed, n_jobs=jobs), 'Rank'),
        (KNOP(pool, random_state=seed, n_jobs=jobs), 'KNOP'),
        (LCA(pool, random_state=seed, n_jobs=jobs), 'LCA'),
        (MLA(pool, random_state=seed, n_jobs=jobs), 'MLA'),
        (APriori(pool, random_state=seed, n_jobs=jobs), 'APriori'),
        (METADES(pool, random_state=seed, n_jobs=jobs), 'METADES'),
    ]


def f1_threshold(labels, probabilities):
    """Match the GCN-Proto-MoE F1 grid search on an internal validation fold."""
    best_f1 = -1
    best_threshold = 0.5
    for threshold in np.linspace(0.01, 0.99, 500):
        predicted = (probabilities >= threshold).astype(int)
        if not predicted.any():
            continue
        score = f1_score(labels, predicted, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold
    return float(best_threshold)


def predict_outputs(selector, scaled):
    predicted = selector.predict(scaled)
    positive_column = np.flatnonzero(selector.classes_ == 1).item()
    probabilities = selector.predict_proba(scaled)[:, positive_column]
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Nonfinite probabilities from {type(selector).__name__}")
    return predicted, probabilities


def input_hashes(args):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.train, args.external]}


def validate_reuse(args, features, hashes):
    if args.reuse_models_from is None:
        return
    previous = json.loads((args.reuse_models_from / 'run_metadata.json').read_text(encoding='utf-8'))
    if previous['status'] != 'complete' or previous['features'] != features:
        raise ValueError('Reusable run is incomplete or has different features')
    if previous['input_sha256'] != hashes:
        raise ValueError('Reusable models were trained with different input files or contents')
    for key in ['seed', 'folds', 'dsel_fraction']:
        if previous['arguments'][key] != getattr(args, key):
            raise ValueError(f'Reusable run has a different {key}')
    for package, version in previous['versions'].items():
        if version is not None and importlib.metadata.version(package) != version:
            raise ValueError(f'Reusable model environment differs: {package}')


def prepare_pool(X, y, development, args, checkpoint):
    pool_rows, dsel_rows = train_test_split(
        development, test_size=args.dsel_fraction, stratify=y[development],
        random_state=args.seed,
    )
    if args.reuse_models_from is not None:
        old = joblib.load(args.reuse_models_from / 'models' / checkpoint)
        scaler = old['scaler']
        # All selectors within a fold share the same DSEL partition; any of
        # them can serve to validate that the reused features/labels match.
        old_selector = next(iter(old['selectors'].values()))
        expected_names = pool_names()
        if old['pool_names'] != expected_names or len(old_selector.pool_classifiers_) != len(expected_names):
            raise ValueError('Reusable classifier definitions do not match the baseline pool')
        if int(scaler.n_samples_seen_) != len(pool_rows):
            raise ValueError('Reusable scaler was fitted on a different training partition')
        dsel = scaler.transform(X[dsel_rows])
        if not np.allclose(old_selector.DSEL_data_, dsel, rtol=0, atol=1e-12):
            raise ValueError('Reusable DSEL features or row order differ')
        if not np.array_equal(old_selector.DSEL_target_, y[dsel_rows]):
            raise ValueError('Reusable DSEL labels differ')
        pool = old_selector.pool_classifiers_
        print(f'  Reused verified {len(expected_names)}-classifier checkpoint: {checkpoint}', flush=True)
    else:
        scaler = StandardScaler().fit(X[pool_rows])
        models = make_models(args.seed)
        for classifier, name in models:
            print(f'  Fitting {name}: {len(pool_rows)} classifier-training records', flush=True)
            classifier.fit(scaler.transform(X[pool_rows]), y[pool_rows])
        pool = [classifier for classifier, _ in models]
        dsel = scaler.transform(X[dsel_rows])
    return scaler, pool, pool_rows, dsel_rows, dsel


def metric_record(method, fold, operating_point, cutoff, y, predicted, probabilities):
    return {'method': method, 'fold': fold, 'operating_point': operating_point,
            'Threshold': cutoff, **metrics(y, predicted, probabilities)}


def write_summary(output, records):
    table = pd.DataFrame(records)
    table.to_csv(output / 'internal_fold_metrics.csv', index=False)
    rows = []
    for (method, operating_point), group in table.groupby(['method', 'operating_point'], sort=False):
        for name in METRIC_COLUMNS:
            rows.append({'method': method, 'operating_point': operating_point,
                         'metric': name, 'mean': group[name].mean(),
                         'SD': np.std(group[name], ddof=0), 'completed_folds': len(group)})
    pd.DataFrame(rows).to_csv(output / 'internal_cv_summary.csv', index=False)


def run(args):
    if not 0 < args.dsel_fraction < 1:
        raise ValueError('DSEL fraction must be between 0 and 1')
    train, external, features, X, y, Xe, ye = load_data(args.train, args.external)
    hashes = input_hashes(args)
    validate_reuse(args, features, hashes)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / 'models').mkdir()
    (output / 'predictions').mkdir()
    method_names = [name for _, name in make_selectors([], args.seed, args.jobs)]
    metadata = {
        'arguments': {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        'versions': {name: importlib.metadata.version(name)
                     for name in ['deslib', 'scikit-learn', 'numpy', 'pandas', 'lightgbm']},
        'methods': method_names, 'features': features, 'n_features': len(features),
        'internal_n': len(y), 'internal_positive': int(y.sum()),
        'external_n': len(ye), 'external_positive': int(ye.sum()),
        'input_sha256': hashes,
        'protocol': {
            'classifiers': 'Fixed pool of eleven baseline classifiers with identical definitions and parameters everywhere',
            'folds': 'Stratified outer folds; 70/30 classifier-training/DSEL split within each development partition',
            'classification': 'Native predictions and internally F1-optimized probability predictions both retained',
            'threshold_search': 'GCN-Proto-MoE grid: 500 evenly spaced cutoffs from 0.01 to 0.99 inclusive; '
                                'probability >= cutoff; skip all-negative predictions; '
                                'keep first maximum F1; default to 0.5 if no cutoff remains',
            'external_threshold': 'Per-method arithmetic mean of internal validation-fold F1-optimal cutoffs; frozen externally',
            'internal_optimized_metrics': 'Exploratory: each validation fold selects and evaluates its cutoff; not nested',
            'scaler': 'Fitted only on classifier-training rows',
            'final_system': 'New 70/30 split of the full internal cohort; no classifier refit on DSEL',
            'reuse': 'Optional immutable checkpoints, checked against input hashes, feature order, partitions and DSEL arrays',
            'GMean': 'sqrt(precision*recall)', 'GPR': 'Alias of GMean, retained for compatibility',
            'unit': 'All original rows retained without deduplication; row-based folds',
        },
        'status': 'started',
    }
    metadata_path = output / 'run_metadata.json'
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'feature_columns.json').write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding='utf-8')
    all_rows = np.arange(len(y))
    oof = {name: {'probability': np.full(len(y), np.nan),
                  'native': np.full(len(y), -1, dtype=int),
                  'optimized': np.full(len(y), -1, dtype=int),
                  'fold': np.zeros(len(y), dtype=int),
                  'threshold': np.full(len(y), np.nan)} for name in method_names}
    cutoffs = {name: [] for name in method_names}
    records, split_roles = [], []
    split_counts = []
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (development, validation) in enumerate(splitter.split(X, y), start=1):
        print(f'Fold {fold}/{args.folds}: all 14 DS/DES methods', flush=True)
        scaler, pool, pool_rows, dsel_rows, Xd = prepare_pool(
            X, y, development, args, f'fold_{fold}.joblib')
        Xv = scaler.transform(X[validation])
        selectors = {}
        for selector, name in make_selectors(pool, args.seed, args.jobs):
            print(f'  Fold {fold}: fitting/evaluating {name}', flush=True)
            selector.fit(Xd, y[dsel_rows])
            native, probabilities = predict_outputs(selector, Xv)
            cutoff = f1_threshold(y[validation], probabilities)
            optimized = (probabilities >= cutoff).astype(int)
            cutoffs[name].append(cutoff)
            oof[name]['probability'][validation] = probabilities
            oof[name]['native'][validation] = native
            oof[name]['optimized'][validation] = optimized
            oof[name]['fold'][validation] = fold
            oof[name]['threshold'][validation] = cutoff
            records.append(metric_record(name, fold, 'native', None, y[validation], native, probabilities))
            records.append(metric_record(name, fold, 'fold_F1_optimized_exploratory', cutoff,
                                         y[validation], optimized, probabilities))
            selectors[name] = selector
        write_summary(output, records)
        for role, indices in [('pool_train', pool_rows), ('DSEL', dsel_rows), ('validation', validation)]:
            split_roles.extend(role_rows(indices, role, fold))
            split_counts.append({'fold': fold, 'role': role, 'N': len(indices),
                                 'DILI': int(y[indices].sum()), 'negative': int((y[indices] == 0).sum())})
        joblib.dump({'scaler': scaler, 'pool_classifiers': pool, 'selectors': selectors,
                     'pool_names': pool_names(),
                     'features': features, 'validation_thresholds': {name: cutoffs[name][-1] for name in method_names}},
                    output / 'models' / f'fold_{fold}.joblib')
        pd.DataFrame(split_roles).to_csv(output / 'internal_split_roles.csv', index=False)
        print(f'  Saved fold {fold}: 14 methods completed', flush=True)

    threshold_rows = []
    frozen = {name: float(np.mean(cutoffs[name])) for name in method_names}
    for name in method_names:
        assert np.isfinite(oof[name]['probability']).all()
        table = pd.DataFrame({
            'row_index': all_rows, 'patient_id': train[PATIENT_ID_COLUMN], 'fold': oof[name]['fold'],
            'y_true': y, 'y_prob': oof[name]['probability'], 'y_pred_native': oof[name]['native'],
            'y_pred': oof[name]['optimized'], 'fold_threshold': oof[name]['threshold'],
        })
        table.to_csv(output / 'predictions' / f'{name}_internal_oof_predictions.csv', index=False, encoding='utf-8-sig')
        threshold_rows.append({'method': name, **{f'fold_{i+1}': cutoff for i, cutoff in enumerate(cutoffs[name])},
                               'external_frozen_threshold': frozen[name]})
    pd.DataFrame(threshold_rows).to_csv(output / 'thresholds.csv', index=False)

    print('Final systems: train DSEL selectors, then evaluate external patients', flush=True)
    scaler, pool, pool_rows, dsel_rows, Xd = prepare_pool(
        X, y, all_rows, args, 'DS_DES_final.joblib')
    Xexternal = scaler.transform(Xe)
    external_records, final_selectors = [], {}
    for selector, name in make_selectors(pool, args.seed, args.jobs):
        print(f'  Final: fitting/evaluating {name}', flush=True)
        selector.fit(Xd, y[dsel_rows])
        native, probabilities = predict_outputs(selector, Xexternal)
        cutoff = frozen[name]
        optimized = (probabilities >= cutoff).astype(int)
        external_records.append(metric_record(name, 0, 'native', None, ye, native, probabilities))
        external_records.append(metric_record(name, 0, 'internal_mean_F1_threshold', cutoff, ye, optimized, probabilities))
        table = pd.DataFrame({'row_index': np.arange(len(ye)), 'patient_id': external[PATIENT_ID_COLUMN],
                              'y_true': ye, 'y_prob': probabilities, 'y_pred_native': native,
                              'y_pred': optimized, 'threshold': cutoff})
        table.to_csv(output / 'predictions' / f'{name}_predictions.csv', index=False, encoding='utf-8-sig')
        np.savez_compressed(output / 'predictions' / f'{name}_predictions.npz', y_true=ye,
                            y_prob=probabilities, y_pred=optimized, y_pred_native=native,
                            row_index=np.arange(len(ye)), patient_id=external[PATIENT_ID_COLUMN].to_numpy(dtype=str))
        final_selectors[name] = selector
        pd.DataFrame(external_records).to_csv(output / 'external_metrics.csv', index=False)
    joblib.dump({'scaler': scaler, 'pool_classifiers': pool, 'selectors': final_selectors,
                 'pool_names': pool_names(),
                 'features': features, 'external_thresholds': frozen}, output / 'models' / 'DS_DES_final.joblib')
    final_roles = role_rows(pool_rows, 'pool_train', 0) + role_rows(dsel_rows, 'DSEL', 0)
    pd.DataFrame(final_roles).to_csv(output / 'final_split_roles.csv', index=False)
    for role, indices in [('pool_train', pool_rows), ('DSEL', dsel_rows)]:
        split_counts.append({'fold': 0, 'role': role, 'N': len(indices),
                             'DILI': int(y[indices].sum()), 'negative': int((y[indices] == 0).sum())})
    pd.DataFrame(split_counts).to_csv(output / 'split_counts.csv', index=False)
    metadata.update({'status': 'complete', 'external_thresholds': frozen})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    result = pd.DataFrame(external_records)
    print(result.loc[result.operating_point == 'internal_mean_F1_threshold',
                     ['method', 'Threshold', 'MCC', 'F1', 'GMean', 'AUROC', 'AUPRC']].to_string(index=False), flush=True)
    print(f'Saved: {output}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True,
                        help='Internal training cohort CSV (preprocessed tabular input)')
    parser.add_argument('--external', type=Path, required=True,
                        help='External validation cohort CSV (preprocessed tabular input)')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'ds_des_results')
    parser.add_argument('--dsel-fraction', type=float, default=0.3)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=-1)
    parser.add_argument('--reuse-models-from', type=Path)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())

