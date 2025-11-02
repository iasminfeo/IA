import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn import metrics

import matplotlib.pyplot as plt
from io import StringIO
from IPython.display import Image, display
import pydotplus

# skopt (otimizador bayesiano)
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


df = pd.read_csv("../base/train.csv")

y = df['Survived'].astype(int)
X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


numeric_features = ['Pclass','Age','SibSp','Parch','Fare']
categorical_features = ['Sex','Embarked']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


tree = DecisionTreeClassifier(random_state=42)

pipe = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', tree)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


search_spaces = {
    'model__criterion': Categorical(['gini','entropy']),  
    'model__splitter': Categorical(['best','random']),
    'model__max_depth': Integer(2, 20),
    'model__min_samples_split': Integer(2, 100),
    'model__min_samples_leaf': Integer(1, 50),
    'model__max_features': Categorical([None, 'sqrt', 'log2', 0.5, 0.7, 0.9]),
    'model__class_weight': Categorical([None, 'balanced']),
    'model__ccp_alpha': Real(1e-6, 5e-2, prior='log-uniform')
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_spaces,
    n_iter=64,                 
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    return_train_score=True,
    refit=True                 


opt.fit(X_train, y_train)

print("Melhores hiperparâmetros (BayesSearchCV):")
print(opt.best_params_)
print(f"ROC-AUC (CV) do melhor: {opt.best_score_:.4f}")


best_model = opt.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- Avaliação no Teste ---")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("ROC-AUC:", metrics.roc_auc_score(y_test, y_proba))
print("Relatório:\n", metrics.classification_report(y_test, y_pred, digits=3))
print("Matriz de confusão:\n", metrics.confusion_matrix(y_test, y_pred))


fitted_pre = best_model.named_steps['preprocess']
feature_names_out = fitted_pre.get_feature_names_out()
tree_best = best_model.named_steps['model']


plt.figure(figsize=(22, 10))
plot_tree(
    tree_best,
    feature_names=feature_names_out,
    class_names=['0','1'],
    filled=True, rounded=True
)
plt.tight_layout()
plt.savefig('titanic_bayes_tree_matplotlib.png', dpi=150)
plt.show()


dot_data = StringIO()
export_graphviz(
    tree_best, out_file=dot_data,
    filled=True, rounded=True, special_characters=True,
    feature_names=feature_names_out,
    class_names=['0','1']
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic_bayes_tree.png')
display(Image(filename='titanic_bayes_tree.png'))