# -*- coding: utf-8 -*-
"""
CART do zero adaptado para o Titanic (train.csv)
- Critério: Gini, splits binários
- Contínuos: limiar ótimo (varre pontos médios)
- Categóricos: split binário por subconjunto (ordenação por taxa de positivos)
- Split 80/20 estratificado (seed configurável)
- Exporta árvore, regras e relatório
"""

import argparse
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import pandas as pd
import os

# =========================
# Nó da árvore (seu estilo)
# =========================
class CARTNode:
    def __init__(
        self,
        atributo: Optional[int] = None,
        nome_atributo: Optional[str] = None,
        limiar: Optional[float] = None,               # usado para numéricos
        cats_esq: Optional[Set[Any]] = None,          # usado para categóricos (lado esquerdo = conjunto de categorias)
        is_categorical: bool = False,
        filhos: Optional[Dict[str, 'CARTNode']] = None,
        rotulo: Optional[Any] = None,
        is_leaf: bool = False
    ):
        self.atributo = atributo
        self.nome_atributo = nome_atributo
        self.limiar = limiar
        self.cats_esq = cats_esq or set()
        self.is_categorical = is_categorical
        self.filhos = filhos or {}   # {'<': filho_esq, '>=': filho_dir} OU {'in': filho_esq, 'out': filho_dir}
        self.rotulo = rotulo
        self.is_leaf = is_leaf

    def __str__(self, nivel=0) -> str:
        ident = "  " * nivel
        if self.is_leaf:
            return f"{ident}Leaf: {self.rotulo}\n"
        if self.is_categorical:
            cond = f"[{self.nome_atributo} ∈ {sorted(list(self.cats_esq))}]"
            s = f"{ident}{cond}\n"
            for cond_lab, filho in self.filhos.items():
                seta = "∈" if cond_lab == 'in' else "∉"
                s += f"{ident}  → {self.nome_atributo} {seta} grupo:\n{filho.__str__(nivel + 2)}"
            return s
        else:
            s = f"{ident}[{self.nome_atributo} < {self.limiar:.6g}]\n"
            for cond_lab, filho in self.filhos.items():
                simb = "<" if cond_lab == '<' else ">="
                s += f"{ident}  → {self.nome_atributo} {simb} {self.limiar:.6g}:\n{filho.__str__(nivel + 2)}"
            return s

    # Extrai regras if-then em texto
    def extract_rules(self, path: Optional[List[str]] = None) -> List[str]:
        path = path or []
        rules: List[str] = []
        if self.is_leaf:
            cond = " AND ".join(path) if path else "(TRUE)"
            rules.append(f"{cond} => predict {self.rotulo}")
            return rules
        if self.is_categorical:
            left_cond = f"{self.nome_atributo} ∈ {{{', '.join(map(str, sorted(list(self.cats_esq))))}}}"
            right_cond = f"{self.nome_atributo} ∉ {{{', '.join(map(str, sorted(list(self.cats_esq))))}}}"
            rules += self.filhos['in'].extract_rules(path + [left_cond])
            rules += self.filhos['out'].extract_rules(path + [right_cond])
            return rules
        else:
            left_cond = f"{self.nome_atributo} < {self.limiar:.6g}"
            right_cond = f"{self.nome_atributo} >= {self.limiar:.6g}"
            rules += self.filhos['<'].extract_rules(path + [left_cond])
            rules += self.filhos['>='].extract_rules(path + [right_cond])
            return rules


# =========================
# Métricas
# =========================
def gini_impureza(y) -> float:
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        return 0.0
    cnt = Counter(y)
    return 1.0 - sum((c / n) ** 2 for c in cnt.values())

def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())

def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [0, 1]
    m = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[labels.index(t), labels.index(p)] += 1
    return m


# =========================
# Melhor divisão (numérica)
# =========================
def best_split_numeric(col: np.ndarray, y: np.ndarray, min_leaf: int) -> Tuple[Optional[float], float]:
    """
    Varre limiares candidatos (midpoints) e retorna (melhor_limiar, gini_min).
    """
    n = len(y)
    vals = col
    uniq = np.unique(vals)
    if uniq.size < 2:
        return None, np.inf

    # candidatos: pontos médios entre valores consecutivos
    thr_cands = (uniq[1:] + uniq[:-1]) / 2.0
    best_thr, best_gini = None, np.inf

    for thr in thr_cands:
        mask_left = vals < thr
        nL, nR = mask_left.sum(), n - mask_left.sum()
        if nL < min_leaf or nR < min_leaf:
            continue
        g = (nL / n) * gini_impureza(y[mask_left]) + (nR / n) * gini_impureza(y[~mask_left])
        if g < best_gini:
            best_gini, best_thr = g, thr
    return best_thr, best_gini
# =========================
# Melhor divisão (categórico)
# =========================
def best_split_categorical(col: np.ndarray, y: np.ndarray, min_leaf: int) -> Tuple[Optional[Set[Any]], float]:
    """
    Estratégia: ordena categorias por taxa de positivos (y==1) e varre
    splits por prefixo dessa ordem. Retorna (conjunto_esquerda, gini_min).
    """
    n = len(y)
    cats, idxs = np.unique(col, return_inverse=True)
    if cats.size < 2:
        return None, np.inf
    # taxa de positivos por categoria
    rates = []
    for i, c in enumerate(cats):
        mask = (idxs == i)
        yi = y[mask]
        rate = (yi == 1).mean() if yi.size > 0 else 0.0
        rates.append((c, rate))
    rates.sort(key=lambda t: t[1])  # ordem crescente de taxa de positivos
    ordered = [c for c, _ in rates]

    best_set, best_gini = None, np.inf
    for k in range(1, len(ordered)):
        left_set = set(ordered[:k])
        mask_left = np.isin(col, list(left_set))
        nL, nR = mask_left.sum(), n - mask_left.sum()
        if nL < min_leaf or nR < min_leaf:
            continue
        g = (nL / n) * gini_impureza(y[mask_left]) + (nR / n) * gini_impureza(y[~mask_left])
        if g < best_gini:
            best_gini, best_set = g, left_set

    return best_set, best_gini


# =========================
# Treinamento CART
# =========================
def majority_class(y):
    return Counter(y).most_common(1)[0][0]

def melhor_divisao_CART(
    X: np.ndarray, y: np.ndarray,
    feature_names: List[str],
    categorical_features: Set[str],
    min_samples_leaf: int
) -> Tuple[Optional[int], Optional[float], Optional[Set[Any]], bool]:
    """
    Retorna:
      (indice_atributo, limiar (numérico ou None), cats_esq (set ou None), is_categorical)
    """
    n, d = X.shape
    best_feat = None
    best_thr = None
    best_cats = None
    best_is_cat = False
    best_gini = np.inf

    for j in range(d):
        name = feature_names[j]
        col = X[:, j]
        if name in categorical_features:
            cats_set, g = best_split_categorical(col, y, min_samples_leaf)
            if g < best_gini:
                best_gini = g
                best_feat = j
                best_thr = None
                best_cats = cats_set
                best_is_cat = True
        else:
            thr, g = best_split_numeric(col, y, min_samples_leaf)
            if g < best_gini:
                best_gini = g
                best_feat = j
                best_thr = thr
                best_cats = None
                best_is_cat = False

    return best_feat, best_thr, best_cats, best_is_cat

def construir_arvore_CART(
    X: np.ndarray, y: np.ndarray,
    feature_names: List[str],
    categorical_features: Set[str],
    profundidade_max: Optional[int] = None,
    profundidade: int = 0,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1
) -> CARTNode:

    # 1) parada: nó puro
    if len(set(y)) == 1:
        return CARTNode(rotulo=y[0], is_leaf=True)

    # 2) sem dados, ou profundidade atingida, ou min_samples_split
    if len(y) == 0 or (profundidade_max is not None and profundidade >= profundidade_max) or len(y) < min_samples_split:
        return CARTNode(rotulo=majority_class(y), is_leaf=True)

    # 3) melhor split
    j, thr, cats_esq, is_cat = melhor_divisao_CART(
        X, y, feature_names, categorical_features, min_samples_leaf
    )
    if j is None:
        return CARTNode(rotulo=majority_class(y), is_leaf=True)

    name = feature_names[j]
    if is_cat:
        mask_left = np.isin(X[:, j], list(cats_esq))
        mask_right = ~mask_left
        if mask_left.sum() < min_samples_leaf or mask_right.sum() < min_samples_leaf:
            return CARTNode(rotulo=majority_class(y), is_leaf=True)
        left_node = construir_arvore_CART(
            X[mask_left], y[mask_left], feature_names, categorical_features,
            profundidade_max, profundidade + 1, min_samples_split, min_samples_leaf
        )
        right_node = construir_arvore_CART(
            X[mask_right], y[mask_right], feature_names, categorical_features,
            profundidade_max, profundidade + 1, min_samples_split, min_samples_leaf
        )
        return CARTNode(
            atributo=j,
            nome_atributo=name,
            cats_esq=cats_esq,
            is_categorical=True,
            filhos={'in': left_node, 'out': right_node},
            rotulo=majority_class(y),
            is_leaf=False
        )
    else:
        mask_left = X[:, j] < thr
        mask_right = ~mask_left
        if mask_left.sum() < min_samples_leaf or mask_right.sum() < min_samples_leaf:
            return CARTNode(rotulo=majority_class(y), is_leaf=True)
        left_node = construir_arvore_CART(
            X[mask_left], y[mask_left], feature_names, categorical_features,
            profundidade_max, profundidade + 1, min_samples_split, min_samples_leaf
        )
        right_node = construir_arvore_CART(
            X[mask_right], y[mask_right], feature_names, categorical_features,
            profundidade_max, profundidade + 1, min_samples_split, min_samples_leaf
        )
        return CARTNode(
            atributo=j,
            nome_atributo=name,
            limiar=thr,
            is_categorical=False,
            filhos={'<': left_node, '>=': right_node},
            rotulo=majority_class(y),
            is_leaf=False
        )

def prever_CART(amostra: np.ndarray, arvore: CARTNode) -> Any:
    node = arvore
    while not node.is_leaf:
        if node.is_categorical:
            if amostra[node.atributo] in node.cats_esq:
                node = node.filhos['in']
            else:
                node = node.filhos['out']
        else:
            if float(amostra[node.atributo]) < float(node.limiar):
                node = node.filhos['<']
            else:
                node = node.filhos['>=']
    return node.rotulo

def avaliar_CART(X: np.ndarray, y: np.ndarray, arvore: CARTNode) -> float:
    preds = [prever_CART(X[i], arvore) for i in range(len(X))]
    return accuracy(y, preds)


# =========================
# Titanic: preparação
# =========================
REQ_COLS = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

def load_and_prepare_titanic(path_csv: str):
    df = pd.read_csv(path_csv)
    df = df[REQ_COLS].copy()

    # Missing: Age/Fare (mediana), Embarked (moda)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

    # Definir quais serão categóricos para CART
    categorical_features = {'Sex', 'Embarked'}   # Pclass como numérico por padrão
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # Converte DataFrame -> matriz numpy mantendo tipos:
    # Para categóricos, manteremos como strings/objetos; numéricos como float
    X_cols = []
    for feat in feature_names:
        if feat in categorical_features:
            X_cols.append(df[feat].astype(object).to_numpy())
        else:
            X_cols.append(df[feat].astype(float).to_numpy())
    X = np.column_stack(X_cols)
    y = df['Survived'].astype(int).to_numpy()

    return X, y, feature_names, categorical_features

def stratified_split(y: np.ndarray, test_size=0.2, seed=42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    test_idx = []
    for c in np.unique(y):
        class_idx = idx[y == c]
        rng.shuffle(class_idx)
        n_test = max(1, int(round(test_size * len(class_idx))))
        test_idx.extend(class_idx[:n_test])
    test_idx = np.array(sorted(test_idx))
    train_idx = np.array([i for i in idx if i not in set(test_idx)])
    return train_idx, test_idx


# =========================
# Persistência das saídas
# =========================
def save_tree_and_rules(tree: CARTNode, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cart_tree.txt"), "w", encoding="utf-8") as f:
        f.write(str(tree))
    with open(os.path.join(out_dir, "cart_rules.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(tree.extract_rules()))

def save_report(out_dir: str, acc_tr: float, acc_te: float, cm_tr, cm_te):
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"""# CART (Gini, binário) — Titanic

## Preparação
- Colunas: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked (target: Survived)
- Missing: Age (mediana), Fare (mediana), Embarked (moda)
- Partição: 80/20 estratificada (seed fixo)

## Resultados
- **Acurácia (Treino)**: **{acc_tr:.4f}**
- **Acurácia (Teste)**:  **{acc_te:.4f}**

### Matrizes de confusão (linhas = real, colunas = predito)
**Treino**
{cm_tr}

**Teste**
{cm_te}
""")


# =========================
# Main / CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Caminho para train.csv do Titanic")
    ap.add_argument("--out", required=True, help="Pasta para salvar saídas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--min_samples_split", type=int, default=2)
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    args = ap.parse_args()

    X, y, feature_names, categorical_features = load_and_prepare_titanic(args.data)
    tr_idx, te_idx = stratified_split(y, 0.2, args.seed)

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    tree = construir_arvore_CART(
        X_tr, y_tr,
        feature_names=feature_names,
        categorical_features=categorical_features,
        profundidade_max=args.max_depth,
        profundidade=0,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf
    )

    # Avaliação
    acc_tr = avaliar_CART(X_tr, y_tr, tree)
    acc_te = avaliar_CART(X_te, y_te, tree)
    cm_tr = confusion_matrix(y_tr, [prever_CART(x, tree) for x in X_tr])
    cm_te = confusion_matrix(y_te, [prever_CART(x, tree) for x in X_te])

    # Saídas
    os.makedirs(args.out, exist_ok=True)
    save_tree_and_rules(tree, args.out)
    save_report(args.out, acc_tr, acc_te, cm_tr, cm_te)

    print("===== CART treinado =====")
    print(tree)
    print(f"ACC treino: {acc_tr:.4f}")
    print(f"ACC teste : {acc_te:.4f}")
    print("Arquivos salvos em:", os.path.abspath(args.out))

if __name__ == "__main__":
    main()