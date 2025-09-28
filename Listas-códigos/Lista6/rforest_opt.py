import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score

from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real

df = pd.read_csv('../base/train.csv')

y = df['Survived']
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Sex', 'Embarked']
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipe = Pipeline(steps=[
    ('prep', preprocessor),
    ('model', RandomForestClassifier(n_jobs=-1, random_state=42))
])

search_spaces = {
    'model__n_estimators': Integer(100, 1000),
    'model__max_depth': Integer(4, 50),
    'model__min_samples_split': Integer(2, 20),
    'model__min_samples_leaf': Integer(1, 10),
    'model__max_features': Categorical(['sqrt', 'log2', 0.5, 0.7]),
    'model__class_weight': Categorical(['balanced', 'balanced_subsample', None])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

bayes = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_spaces,
    n_iter=50,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

bayes.fit(X_train, y_train)

print("\n=== Melhor combinação de hiperparâmetros ===")
print(bayes.best_params_)
print(f"Melhor score CV ({bayes.scoring}): {bayes.best_score_:.4f}")

# --- Avaliação no conjunto de teste ---
best_clf = bayes.best_estimator_
y_pred = best_clf.predict(X_test)
y_prob = best_clf.predict_proba(X_test)[:, 1]

print("\n=== Métricas no conjunto de teste (Random Forest Otimizado) ===")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", values_format="d")
plt.title("Matriz de Confusão (Random Forest Otimizado)")
plt.show()