import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from io import StringIO
from IPython.display import Image, display
import pydotplus
import matplotlib.pyplot as plt

df = pd.read_csv("../base/train.csv")

# Campos com muitos NaNs no Titanic: Age, Embarked, Fare (em alguns casos)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

#Descartando Name, Ticket e Cabin no baseline
base_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_dum = pd.get_dummies(df[base_cols], columns=['Sex', 'Embarked'], drop_first=False)
y = df['Survived'].astype(int)

feature_cols = X_dum.columns.tolist()
X = X_dum

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(
    criterion='gini',        
    random_state=42,
    max_depth=4,
    min_samples_leaf=5
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision/Recall/F1 por classe:\n", metrics.classification_report(y_test, y_pred, digits=3))
print("ROC-AUC:", metrics.roc_auc_score(y_test, y_proba))

cm = metrics.confusion_matrix(y_test, y_pred)
print("Matriz de confusão:\n", cm)

dot_data = StringIO()
export_graphviz(
    clf, out_file=dot_data,
    filled=True, rounded=True, special_characters=True,
    feature_names=feature_cols,
    class_names=['0','1']
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('titanic_tree.png')  
display(Image(filename='titanic_tree.png'))

plt.figure(figsize=(18, 8))
plot_tree(
    clf, feature_names=feature_cols, class_names=['0','1'],
    filled=True, rounded=True
)
plt.tight_layout()
plt.savefig('titanic_tree_matplotlib.png', dpi=150)
plt.show()