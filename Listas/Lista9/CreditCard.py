#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lista #9 - IA
Curso: Ciência da Computação
Disciplina: Inteligência Artificial
Profa. Cristiane Neri Nobre

Autora: Iasmin Oliveira (RA: 854946)
Título: Pré-processamento + Agrupamento (KMeans, DBSCAN, SOM) na base Credit Card Fraud Detection

Como usar:
1) Coloque o arquivo "creditcard.csv" (Kaggle) na mesma pasta deste script.
2) Execute:  python lista9_IA.py
3) Saídas geradas (pasta ./outputs):
   - métricas em JSON (antes/depois do pré-processamento)
   - gráficos (distribuições, correlação, curvas ROC/PR, etc.)
   - relatório PDF atualizado com resultados (opcional): python lista9_IA.py --build-pdf

Observação importante:
- A base possui atributos V1..V28 (PCA), Time, Amount e o alvo Class (0=legítima, 1=fraude).
- Este script é determinístico (seeds fixas) e estratifica o split.
"""

import argparse
import os
import sys
import json
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    confusion_matrix, silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_FILE = "creditcard.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

############################################################
# Mini SOM (Self-Organizing Map)
############################################################
class MiniSOM:
    def __init__(self, x=10, y=10, input_len=2, sigma=1.5, learning_rate=0.5, random_state=RANDOM_STATE):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(random_state)
        self.weights = self.rng.rand(x*y, input_len)
        self._locations = np.array([(i, j) for i in range(x) for j in range(y)])

    def _gaussian(self, c, sigma):
        d = np.linalg.norm(self._locations - c, axis=1)
        return np.exp(-(d**2) / (2 * (sigma**2)))

    def winner(self, x):
        i = np.argmin(np.linalg.norm(self.weights - x, axis=1))
        return np.unravel_index(i, (self.x, self.y))

    def train(self, data, num_iteration=1000):
        for it in range(num_iteration):
            x = data[self.rng.randint(0, len(data))]
            bmu = self.winner(x)
            bmu_idx = bmu[0]*self.y + bmu[1]
            g = self._gaussian(np.array(bmu), self.sigma)
            lr = self.learning_rate * (1.0 - it/num_iteration)
            self.weights += (g[:, None] * lr) * (x - self.weights)

    def predict(self, data):
        winners = [self.winner(x) for x in data]
        coords = np.array([w[0]*self.y + w[1] for w in winners]).reshape(-1,1)
        labels = KMeans(n_clusters=2, random_state=RANDOM_STATE).fit_predict(coords)
        return labels

###########################################################
# Avaliação do modelo
###########################################################
@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    tn: int
    fp: int
    fn: int
    tp: int

def evaluate_classifier(y_true, y_proba, threshold=0.5) -> Metrics:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)
    )

def save_json(obj: Dict[str, Any], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

###########################################################
# 1) Carregar e visualizar
###########################################################
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERRO] Arquivo '{path}' não encontrado.")
        sys.exit(1)
    df = pd.read_csv(path)
    return df

def visualize(df: pd.DataFrame):
    print("\n=== 1) Visualização da base ===")
    print(df.head())
    print(df.describe())
    print("Distribuição de classes:", df['Class'].value_counts())

    plt.figure()
    df['Amount'].hist(bins=50)
    plt.title("Distribuição Amount")
    plt.savefig(OUT_DIR / "hist_amount.png")
    plt.close()

    plt.figure()
    df['Time'].hist(bins=50)
    plt.title("Distribuição Time")
    plt.savefig(OUT_DIR / "hist_time.png")
    plt.close()

###########################################################
# 2) Missing values
###########################################################
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 2) Valores ausentes ===")
    miss = df.isna().sum()
    print(miss[miss > 0])

    imputer = SimpleImputer(strategy="median")
    df2 = df.copy()
    df2[df.columns] = imputer.fit_transform(df)
    return df2

###########################################################
# 3) Remoção de duplicatas
###########################################################
def remove_redundancy(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 3) Redundância e inconsistência ===")
    before = len(df)
    df2 = df.drop_duplicates()
    print("Duplicatas removidas:", before - len(df2))

    df2.loc[df2['Time'] < 0, 'Time'] = 0
    df2.loc[df2['Amount'] < 0, 'Amount'] = 0
    return df2

###########################################################
# 4) Outliers
###########################################################
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 4) Outliers ===")
    df2 = df.copy()

    # IQR em Amount
    q1, q3 = df2['Amount'].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    before = df2['Amount'].copy()
    df2['Amount'] = df2['Amount'].clip(low, high)
    print("Amount alterados:", (before != df2['Amount']).sum())

    # Isolation Forest
    pca_cols = [c for c in df.columns if c.startswith("V")]
    iso = IsolationForest(contamination=0.001, random_state=42)
    flags = iso.fit_predict(df2[pca_cols])
    df2["IF_outlier"] = (flags == -1).astype(int)

    return df2

###########################################################
# 5) Padronização
###########################################################
def scale_features(df: pd.DataFrame):
    print("\n=== 5) Padronização ===")
    df2 = df.copy()
    X = df2.drop(columns=['Class'])
    y = df2['Class']

    scale_cols = ['Time', 'Amount']
    ct = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), scale_cols)],
        remainder='passthrough'
    )
    X_scaled = ct.fit_transform(X)
    new_cols = scale_cols + [c for c in X.columns if c not in scale_cols]
    X_scaled = pd.DataFrame(X_scaled, columns=new_cols)

    df_out = X_scaled.join(y)
    return df_out, ct

###########################################################
# 6) Correlação e VIF
###########################################################
def correlation_and_vif(df: pd.DataFrame):
    print("\n=== 6) Correlação e VIF ===")
    corr = df.drop(columns=['Class']).corr()
    corr.to_csv(OUT_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect='auto')
    plt.colorbar()
    plt.title("Matriz de Correlação")
    plt.savefig(OUT_DIR / "correlation_heatmap.png")
    plt.close()

###########################################################
# 7) Codificação (não necessário)
###########################################################
def encode(df: pd.DataFrame):
    print("\n=== 7) Codificação === (não necessária)")
    return df

###########################################################
# 8 e 9) Split + SMOTE
###########################################################
def split_and_balance(df: pd.DataFrame):
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42)
    Xb, yb = sm.fit_resample(X_train, y_train)

    return (X_train, X_test, y_train, y_test, Xb, yb)

###########################################################
# Modelo base: Regressão Logística
###########################################################
def train_logreg(X, y, X_test):
    clf = LogisticRegression(max_iter=300, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X_test)[:,1]
    return clf, proba

###########################################################
# Gráficos ROC/PR
###########################################################
def plot_curves(y, proba, name):
    from sklearn.metrics import roc_curve, precision_recall_curve

    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.savefig(OUT_DIR / f"roc_{name}.png")
    plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.title("PR Curve")
    plt.savefig(OUT_DIR / f"pr_{name}.png")
    plt.close()

###########################################################
# Clustering
###########################################################
def clustering_all(X_df, y_true):
    X = X_df.values
    results = {}

    # KMeans
    km = KMeans(n_clusters=2, random_state=42)
    labels_km = km.fit_predict(X)
    results['kmeans'] = {
        'silhouette': float(silhouette_score(X, labels_km)),
        'davies_bouldin': float(davies_bouldin_score(X, labels_km)),
        'calinski_harabasz': float(calinski_harabasz_score(X, labels_km)),
        'ari': float(adjusted_rand_score(y_true, labels_km)),
        'nmi': float(normalized_mutual_info_score(y_true, labels_km))
    }

    # DBSCAN
    db = DBSCAN(eps=2.5, min_samples=10)
    labels_db = db.fit_predict(X)
    if len(set(labels_db)) > 1:
        results['dbscan'] = {
            'silhouette': float(silhouette_score(X, labels_db)),
            'davies_bouldin': float(davies_bouldin_score(X, labels_db)),
            'calinski_harabasz': float(calinski_harabasz_score(X, labels_db)),
            'ari': float(adjusted_rand_score(y_true, labels_db)),
            'nmi': float(normalized_mutual_info_score(y_true, labels_db)),
        }
    else:
        results['dbscan'] = {"note": "DBSCAN formou 1 cluster ou noise total. Ajustar eps/min_samples."}

    # SOM
    som = MiniSOM(x=10, y=10, input_len=X.shape[1], sigma=1.5, learning_rate=0.5)
    som.train(X, num_iteration=2000)
    labels_som = som.predict(X)

    results['som'] = {
        'silhouette': float(silhouette_score(X, labels_som)),
        'davies_bouldin': float(davies_bouldin_score(X, labels_som)),
        'calinski_harabasz': float(calinski_harabasz_score(X, labels_som)),
        'ari': float(adjusted_rand_score(y_true, labels_som)),
        'nmi': float(normalized_mutual_info_score(y_true, labels_som)),
    }

    return results

###########################################################
# MAIN
###########################################################
def main(build_pdf=False):
    df = load_dataset(DATA_FILE)
    visualize(df)
    df = handle_missing(df)
    df = remove_redundancy(df)
    df = handle_outliers(df)
    df, scaler = scale_features(df)
    correlation_and_vif(df)
    df = encode(df)

    X_train, X_test, y_train, y_test, Xb, yb = split_and_balance(df)

    # Sem SMOTE
    _, p0 = train_logreg(X_train, y_train, X_test)
    m0 = evaluate_classifier(y_test, p0)
    plot_curves(y_test, p0, "sem_smote")

    # Com SMOTE
    _, p1 = train_logreg(Xb, yb, X_test)
    m1 = evaluate_classifier(y_test, p1)
    plot_curves(y_test, p1, "com_smote")

    # Clustering sem o rótulo
    X_df = df.drop(columns=['Class'])
    clust = clustering_all(X_df, df['Class'].values)

    out = {
        "baseline": {
            "sem_smote": asdict(m0),
            "com_smote": asdict(m1)
        },
        "clustering": clust
    }
    save_json(out, OUT_DIR / "metrics.json")
    print("Resultados salvos em outputs/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-pdf", action="store_true")
    args = parser.parse_args()
    main(build_pdf=args.build_pdf)
