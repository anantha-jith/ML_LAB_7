from __future__ import annotations
import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)

# Optional libs
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
    HAS_CAT = True
except Exception:
    HAS_CAT = False

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =====================
# Config
# =====================
DATA_PATH = 'ecg_eeg_features.csv.xlsx'  # Update if needed
TARGET_COL: Optional[str] = None          # e.g., 'label' | 'target' | None to auto-detect
TASK: str = 'auto'                        # 'classification' | 'regression' | 'clustering' | 'auto'
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
N_ITER = 30                               # RandomizedSearch iterations per model
N_JOBS = -1
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# Utilities
# =====================

def read_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(path)
    elif path.lower().endswith('.csv'):
        return pd.read_csv(path)
    else:
        # Try reading as CSV first, then Excel
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_excel(path)


def auto_detect_target(df: pd.DataFrame) -> Optional[str]:
    # Heuristics: prefer columns named like label/target/y/class, else last column if it looks categorical
    candidates = [
        'label', 'target', 'y', 'class', 'Class', 'Label', 'Target'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: last column
    return df.columns[-1]


def infer_task(y: Optional[pd.Series]) -> str:
    if y is None:
        return 'clustering'
    if pd.api.types.is_numeric_dtype(y):
        # If numeric with few unique values, treat as classification
        nunique = y.nunique(dropna=True)
        if nunique <= 15:
            return 'classification'
        return 'regression'
    else:
        return 'classification'


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    pre = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    return pre, num_cols, cat_cols


# =====================
# Model Spaces & Param Grids
# =====================

def classification_models_and_params() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    models: Dict[str, Tuple[Any, Dict[str, list]]] = {}
    # SVM
    models['SVM'] = (
        SVC(probability=True, random_state=RANDOM_STATE),
        {
            'clf__C': np.logspace(-2, 2, 20).tolist(),
            'clf__gamma': np.logspace(-4, 0, 20).tolist(),
            'clf__kernel': ['rbf', 'poly', 'sigmoid']
        }
    )
    # Decision Tree
    models['DecisionTree'] = (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            'clf__max_depth': [None] + list(range(2, 31)),
            'clf__min_samples_split': list(range(2, 21)),
            'clf__min_samples_leaf': list(range(1, 21))
        }
    )
    # Random Forest
    models['RandomForest'] = (
        RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300),
        {
            'clf__max_depth': [None] + list(range(2, 31)),
            'clf__min_samples_split': list(range(2, 21)),
            'clf__min_samples_leaf': list(range(1, 11))
        }
    )
    # AdaBoost
    models['AdaBoost'] = (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {
            'clf__n_estimators': list(range(50, 501, 50)),
            'clf__learning_rate': np.linspace(0.01, 1.0, 20).tolist()
        }
    )
    # Naive Bayes
    models['NaiveBayes'] = (
        GaussianNB(),
        {
            # No hyperparams for GaussianNB that need tuning in this grid
        }
    )
    # MLP
    models['MLP'] = (
        MLPClassifier(max_iter=500, random_state=RANDOM_STATE),
        {
            'clf__hidden_layer_sizes': [(64,), (128,), (64,32), (128,64)],
            'clf__alpha': np.logspace(-5, -1, 5).tolist(),
            'clf__learning_rate_init': np.logspace(-4, -2, 5).tolist()
        }
    )
    # XGBoost
    if HAS_XGB:
        models['XGBoost'] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                objective='multi:softprob',
                eval_metric='mlogloss',
                tree_method='hist'
            ),
            {
                'clf__n_estimators': list(range(100, 601, 100)),
                'clf__max_depth': list(range(3, 11)),
                'clf__learning_rate': np.linspace(0.01, 0.3, 10).tolist(),
                'clf__subsample': np.linspace(0.6, 1.0, 5).tolist(),
                'clf__colsample_bytree': np.linspace(0.6, 1.0, 5).tolist()
            }
        )
    # CatBoost
    if HAS_CAT:
        models['CatBoost'] = (
            CatBoostClassifier(random_state=RANDOM_STATE, verbose=False),
            {
                'clf__depth': list(range(4, 11)),
                'clf__learning_rate': np.linspace(0.01, 0.3, 10).tolist(),
                'clf__iterations': list(range(200, 801, 100))
            }
        )
    return models


def regression_models_and_params() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    models: Dict[str, Tuple[Any, Dict[str, list]]] = {}
    # SVR
    models['SVR'] = (
        SVR(),
        {
            'clf__C': np.logspace(-2, 2, 20).tolist(),
            'clf__gamma': np.logspace(-4, 0, 20).tolist(),
            'clf__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }
    )
    # Decision Tree
    models['DecisionTreeReg'] = (
        DecisionTreeRegressor(random_state=RANDOM_STATE),
        {
            'clf__max_depth': [None] + list(range(2, 31)),
            'clf__min_samples_split': list(range(2, 21)),
            'clf__min_samples_leaf': list(range(1, 21))
        }
    )
    # Random Forest
    models['RandomForestReg'] = (
        RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=300),
        {
            'clf__max_depth': [None] + list(range(2, 31)),
            'clf__min_samples_split': list(range(2, 21)),
            'clf__min_samples_leaf': list(range(1, 11))
        }
    )
    # AdaBoost
    models['AdaBoostReg'] = (
        AdaBoostRegressor(random_state=RANDOM_STATE),
        {
            'clf__n_estimators': list(range(50, 501, 50)),
            'clf__learning_rate': np.linspace(0.01, 1.0, 20).tolist()
        }
    )
    # MLPRegressor
    models['MLPReg'] = (
        MLPRegressor(max_iter=500, random_state=RANDOM_STATE),
        {
            'clf__hidden_layer_sizes': [(64,), (128,), (64,32), (128,64)],
            'clf__alpha': np.logspace(-5, -1, 5).tolist(),
            'clf__learning_rate_init': np.logspace(-4, -2, 5).tolist()
        }
    )
    # XGBoost
    if HAS_XGB:
        models['XGBReg'] = (
            XGBRegressor(random_state=RANDOM_STATE, tree_method='hist'),
            {
                'clf__n_estimators': list(range(200, 801, 100)),
                'clf__max_depth': list(range(3, 11)),
                'clf__learning_rate': np.linspace(0.01, 0.3, 10).tolist(),
                'clf__subsample': np.linspace(0.6, 1.0, 5).tolist(),
                'clf__colsample_bytree': np.linspace(0.6, 1.0, 5).tolist()
            }
        )
    # CatBoost
    if HAS_CAT:
        models['CatBoostReg'] = (
            CatBoostRegressor(random_state=RANDOM_STATE, verbose=False),
            {
                'clf__depth': list(range(4, 11)),
                'clf__learning_rate': np.linspace(0.01, 0.3, 10).tolist(),
                'clf__iterations': list(range(200, 801, 100))
            }
        )
    return models


# =====================
# Training / Evaluation Routines
# =====================

def evaluate_classification(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    # ROC-AUC if binary or multiclass with probas
    try:
        if y_proba is not None:
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                # binary
                proba_pos = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, proba_pos)
            else:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
    except Exception:
        pass
    return metrics


def run_classification(X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    best_models: Dict[str, Any] = {}

    for name, (estimator, param_grid) in classification_models_and_params().items():
        pipe = Pipeline([
            ('pre', pre),
            ('clf', estimator)
        ])

        if len(param_grid) > 0:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                n_iter=min(N_ITER, np.prod([len(v) for v in param_grid.values()])),
                scoring='f1_macro',
                cv=CV_FOLDS,
                verbose=0,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
                refit=True
            )
        else:
            # No hyperparams to tune; wrap in identity search
            search = pipe

        print(f"\nTraining {name}...")
        if isinstance(search, RandomizedSearchCV):
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
            best_params = search.best_params_
            cv_score = search.best_score_
        else:
            search.fit(X_train, y_train)
            best_est = search
            best_params = {}
            cv_score = np.nan

        # Evaluate
        y_pred_train = best_est.predict(X_train)
        y_pred_test = best_est.predict(X_test)
        y_proba_test = None
        try:
            y_proba_test = best_est.predict_proba(X_test)
        except Exception:
            try:
                y_proba_test = best_est.decision_function(X_test)
            except Exception:
                pass

        train_metrics = evaluate_classification(y_train, y_pred_train)
        test_metrics = evaluate_classification(y_test, y_pred_test, y_proba_test)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'{name} — Confusion Matrix (Test)')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'confmat_{name}.png'), dpi=150)
        plt.close()

        # Store
        row = {
            'model': name,
            'cv_f1_macro': cv_score,
            'train_accuracy': train_metrics.get('accuracy'),
            'train_f1_macro': train_metrics.get('f1_macro'),
            'test_accuracy': test_metrics.get('accuracy'),
            'test_precision_macro': test_metrics.get('precision_macro'),
            'test_recall_macro': test_metrics.get('recall_macro'),
            'test_f1_macro': test_metrics.get('f1_macro'),
            'test_roc_auc': test_metrics.get('roc_auc', np.nan),
            'test_roc_auc_ovr': test_metrics.get('roc_auc_ovr', np.nan),
            'best_params': json.dumps(best_params)
        }
        results.append(row)
        best_models[name] = best_est

    df_results = pd.DataFrame(results).sort_values('test_f1_macro', ascending=False)
    df_results.to_csv(os.path.join(OUT_DIR, 'classification_results.csv'), index=False)

    # Save top model for explainability
    if not df_results.empty:
        top_name = df_results.iloc[0]['model']
        with open(os.path.join(OUT_DIR, 'best_classification_model.txt'), 'w') as f:
            f.write(str(top_name))
    return df_results


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def run_regression(X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []
    for name, (estimator, param_grid) in regression_models_and_params().items():
        pipe = Pipeline([
            ('pre', pre),
            ('clf', estimator)
        ])

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=min(N_ITER, np.prod([len(v) for v in param_grid.values()])),
            scoring='neg_root_mean_squared_error',
            cv=CV_FOLDS,
            verbose=0,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            refit=True
        )

        print(f"\nTraining {name}...")
        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        best_params = search.best_params_
        cv_score = search.best_score_

        y_pred_train = best_est.predict(X_train)
        y_pred_test = best_est.predict(X_test)

        train_metrics = evaluate_regression(y_train, y_pred_train)
        test_metrics = evaluate_regression(y_test, y_pred_test)

        row = {
            'model': name,
            'cv_neg_rmse': cv_score,
            'train_rmse': train_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'test_rmse': test_metrics['rmse'],
            'test_mae': test_metrics['mae'],
            'test_mape': test_metrics['mape'],
            'test_r2': test_metrics['r2'],
            'best_params': json.dumps(best_params)
        }
        results.append(row)

    df_results = pd.DataFrame(results).sort_values('test_rmse', ascending=True)
    df_results.to_csv(os.path.join(OUT_DIR, 'regression_results.csv'), index=False)
    return df_results


def run_clustering(X: pd.DataFrame) -> pd.DataFrame:
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.preprocessing import StandardScaler

    # Numeric only and scale
    X_num = X.select_dtypes(include=[np.number]).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    results = []

    # KMeans: try a few k
    for k in [2, 3, 4, 5, 6, 8, 10]:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        labels = km.fit_predict(X_scaled)
        row = {
            'model': f'KMeans(k={k})',
            'silhouette': silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan,
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan,
            'davies_bouldin': davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
        }
        results.append(row)

    # Agglomerative: average linkage
    for k in [2, 3, 4, 5, 6, 8, 10]:
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = agg.fit_predict(X_scaled)
        row = {
            'model': f'Agglomerative(k={k})',
            'silhouette': silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan,
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan,
            'davies_bouldin': davies_bouldin_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
        }
        results.append(row)

    # DBSCAN: epsilon sweep
    for eps in [0.3, 0.5, 0.7, 1.0]:
        for min_samples in [3, 5, 10]:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            # Skip if all noise or all one cluster
            if len(set(labels)) <= 1:
                sil = np.nan
                ch = np.nan
                dbi = np.nan
            else:
                sil = silhouette_score(X_scaled, labels)
                ch = calinski_harabasz_score(X_scaled, labels)
                dbi = davies_bouldin_score(X_scaled, labels)
            row = {
                'model': f'DBSCAN(eps={eps},min_samples={min_samples})',
                'silhouette': sil,
                'calinski_harabasz': ch,
                'davies_bouldin': dbi
            }
            results.append(row)

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUT_DIR, 'clustering_results.csv'), index=False)
    return df_results


# =====================
# Explainability (SHAP & LIME)
# =====================

def shap_explain(best_estimator: Any, X_sample: pd.DataFrame, task: str, pre: ColumnTransformer, model_name: str):
    try:
        import shap  # type: ignore
    except Exception as e:
        print("SHAP not installed — skipping explainability.")
        return

    # Transform features as model sees them
    X_trans = pre.fit_transform(X_sample)

    # Try to get underlying model (last step named 'clf')
    model = best_estimator.named_steps.get('clf', best_estimator)

    try:
        explainer = shap.Explainer(model, X_trans)
        shap_values = explainer(X_trans)
        # Summary plot
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'shap_bar_{task}_{model_name}.png'), dpi=150)
        plt.close()

        plt.figure()
        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'shap_beeswarm_{task}_{model_name}.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"SHAP failed: {e}")


def lime_explain(best_estimator: Any, X_train: pd.DataFrame, pre: ColumnTransformer, task: str, model_name: str):
    try:
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore
    except Exception:
        print("LIME not installed — skipping explainability.")
        return

    # Fit the preprocessor
    pre_fitted = pre.fit(X_train)
    X_train_trans = pre_fitted.transform(X_train)

    feature_names = []
    try:
        # Attempt to extract feature names after ColumnTransformer
        ohe = [t for t in pre.transformers_ if t[0] == 'cat'][0][1].named_steps['onehot']
        num_cols = [t for t in pre.transformers_ if t[0] == 'num'][0][2]
        cat_cols = [t for t in pre.transformers_ if t[0] == 'cat'][0][2]
        ohe_names = ohe.get_feature_names_out(cat_cols)
        feature_names = list(num_cols) + list(ohe_names)
    except Exception:
        feature_names = [f'f{i}' for i in range(X_train_trans.shape[1])]

    mode = 'classification' if task == 'classification' else 'regression'

    explainer = LimeTabularExplainer(
        X_train_trans.astype(float),
        feature_names=feature_names,
        mode=mode,
        verbose=False
    )

    # Build prediction function on transformed space
    model = best_estimator.named_steps.get('clf', best_estimator)

    def predict_fn(arr):
        # Model expects transformed input
        if task == 'classification':
            try:
                return model.predict_proba(arr)
            except Exception:
                # fall back to decision_function -> probabilities like outputs (approx)
                vals = model.decision_function(arr)
                if vals.ndim == 1:
                    vals = np.vstack([1-vals, vals]).T
                return vals
        else:
            return model.predict(arr)

    # Explain the first instance
    idx = 0
    exp = explainer.explain_instance(X_train_trans[idx], predict_fn, num_features=min(10, X_train_trans.shape[1]))
    out_path = os.path.join(OUT_DIR, f'lime_{task}_{model_name}.html')
    exp.save_to_file(out_path)


# =====================
# Main
# =====================

def main():
    df = read_dataset(DATA_PATH)
    # Basic cleaning
    df = df.dropna(how='all')

    target = TARGET_COL if TARGET_COL in df.columns if TARGET_COL else None
    if target is None and df.shape[1] >= 2:
        target = auto_detect_target(df)

    print(f"Detected/Using target column: {target}")

    if target is not None and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
    else:
        X = df.copy()
        y = None

    # Decide task
    task = TASK if TASK in {'classification','regression','clustering'} else infer_task(y)
    print(f"Resolved task: {task}")

    # Preprocessor
    pre, num_cols, cat_cols = build_preprocessor(X)

    if task == 'classification':
        results = run_classification(X, y, pre)
        print("\nClassification results (top rows):\n", results.head())
        # Explain top model
        try:
            top_model_name = results.iloc[0]['model']
            # Refit the top model on all data for explanations
            est_tuple = classification_models_and_params()[top_model_name]
            best_pipe = Pipeline([('pre', pre), ('clf', est_tuple[0])])
            best_pipe.fit(X, y)
            shap_explain(best_pipe, X.sample(min(200, len(X)), random_state=RANDOM_STATE), 'classification', pre, top_model_name)
            lime_explain(best_pipe, X, pre, 'classification', top_model_name)
        except Exception as e:
            print(f"Explainability step skipped: {e}")

    elif task == 'regression':
        results = run_regression(X, y, pre)
        print("\nRegression results (top rows):\n", results.head())
        # Explain top model
        try:
            top_model_name = results.iloc[0]['model']
            est_tuple = regression_models_and_params()[top_model_name]
            best_pipe = Pipeline([('pre', pre), ('clf', est_tuple[0])])
            best_pipe.fit(X, y)
            shap_explain(best_pipe, X.sample(min(200, len(X)), random_state=RANDOM_STATE), 'regression', pre, top_model_name)
            lime_explain(best_pipe, X, pre, 'regression', top_model_name)
        except Exception as e:
            print(f"Explainability step skipped: {e}")

    else:  # clustering
        results = run_clustering(X)
        print("\nClustering results (top rows):\n", results.head())

    # Save a quick preview CSV
    df.head(20).to_csv(os.path.join(OUT_DIR, 'data_preview.csv'), index=False)
    print(f"\nAll done. Outputs saved to: {os.path.abspath(OUT_DIR)}")


if __name__ == '__main__':
    main()
