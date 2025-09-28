import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from skopt.space import Real

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
    ('preprocessor', preprocessor),
    ('model', GaussianNB())
])

# --- Otimização Bayesiana ---
search_space = {
    "model__var_smoothing": Real(1e-12, 1e-3, prior="log-uniform")
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

bayes_cv = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_space,
    n_iter=50,
    cv=cv,
    scoring="roc_auc", 
    n_jobs=-1,
    random_state=42,
    refit=True
)

bayes_cv.fit(X_train, y_train)

print("\nMelhores hiperparâmetros encontrados:", bayes_cv.best_params_)
print(f"Melhor ROC-AUC em CV: {bayes_cv.best_score_:.4f}")

best_model = bayes_cv.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report (Teste - Naive Bayes Otimizado):\n", classification_report(y_test, y_pred, digits=3))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Matriz de Confusão - GaussianNB Otimizado no Titanic")
plt.show()