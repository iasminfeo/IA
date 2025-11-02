import pandas as pd
import numpy as np

# Função para calcular a entropia
def calcular_entropia(y):
    valores, contagens = np.unique(y, return_counts=True)
    prob = contagens / len(y)
    return -np.sum(prob * np.log2(prob))

# Função para calcular o ganho de informação
def ganho_informacao(X_coluna, y):
    entropia_total = calcular_entropia(y)
    valores, contagens = np.unique(X_coluna, return_counts=True)
    
    entropia_condicional = 0
    for valor, contagem in zip(valores, contagens):
        y_sub = y[X_coluna == valor]
        entropia_condicional += (contagem / len(X_coluna)) * calcular_entropia(y_sub)
    
    return entropia_total - entropia_condicional

# Função principal do ID3
def ID3(X, y, atributos):
    # Caso base: todos os rótulos são iguais
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    
    # Caso base: sem atributos restantes
    if len(atributos) == 0:
        return np.bincount(y).argmax()
    
    # Escolher o melhor atributo
    ganhos = [ganho_informacao(X[atributo], y) for atributo in atributos]
    melhor_atributo = atributos[np.argmax(ganhos)]
    
    # Criar o nó da árvore
    arvore = {melhor_atributo: {}}
    
    for valor in np.unique(X[melhor_atributo]):
        sub_X = X[X[melhor_atributo] == valor]
        sub_y = y[X[melhor_atributo] == valor]
        sub_atributos = [a for a in atributos if a != melhor_atributo]
        
        arvore[melhor_atributo][valor] = ID3(sub_X, sub_y, sub_atributos)
    
    return arvore

# ============================
# Leitura do arquivo CSV
# ============================

df = pd.read_csv('../test.csv')

# Supondo que a última coluna seja o rótulo
atributos = list(df.columns[:-1])
rotulos = df[df.columns[-1]].values

# Construção da árvore de decisão
arvore_decisao = ID3(df[atributos], rotulos, atributos)

# Exibição da árvore gerada
print("Árvore de decisão gerada:")
print(arvore_decisao)