import numpy as np
import matplotlib.pyplot as plt
import fitz  # PyMuPDF

# Dados extraídos da imagem: 10 amostras de entrada (7 segmentos) e saída one-hot com 4 classes
X = np.array([
    [1,1,1,1,1,1,0],  
    [0,1,1,0,0,0,0],  
    [1,1,0,1,1,0,1],  
    [1,1,1,1,0,0,1], 
    [0,1,1,0,0,1,1],  
    [1,0,1,1,0,1,1],  
    [1,0,1,1,1,1,1],  
    [1,1,1,0,0,0,0],  
    [1,1,1,1,1,1,1], 
    [1,1,1,1,0,1,1]   
])

y = np.array([
    [1,0,0,0],  # classe 0
    [0,1,0,0],  # classe 1
    [0,0,1,0],  # classe 2
    [0,0,0,1],  # classe 3
    [1,0,0,0],  # classe 4
    [0,1,0,0],  # classe 5
    [0,0,1,0],  # classe 6
    [0,0,0,1],  # classe 7
    [1,0,0,0],  # classe 8
    [0,1,0,0]   # classe 9
])

INPUT_LAYER_SIZE = 7
HIDDEN_LAYER_SIZE = 5
OUTPUT_LAYER_SIZE = 4
EPOCHS = 1000
LEARNING_RATE = 0.5

# Funções de ativação
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# Inicialização dos pesos e bias
W1 = np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * np.sqrt(2.0 / INPUT_LAYER_SIZE)
B1 = np.ones((1, HIDDEN_LAYER_SIZE))

W2 = np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * np.sqrt(2.0 / HIDDEN_LAYER_SIZE)
B2 = np.ones((1, OUTPUT_LAYER_SIZE))

# Treinamento
errors = []
for epoch in range(EPOCHS):
    Z1 = np.dot(X, W1) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + B2
    yHat = sigmoid(Z2)

    error = y - yHat
    loss = np.mean(np.square(error))
    errors.append(loss)

    dZ2 = error * sigmoid_derivative(yHat)
    dW2 = np.dot(A1.T, dZ2)
    dB2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1)
    dB1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 += LEARNING_RATE * dW2
    B2 += LEARNING_RATE * dB2
    W1 += LEARNING_RATE * dW1
    B1 += LEARNING_RATE * dB1

# Teste com ruído
X_ruido = X + np.random.normal(0, 0.1, X.shape)
Z1_ruido = np.dot(X_ruido, W1) + B1
A1_ruido = sigmoid(Z1_ruido)
Z2_ruido = np.dot(A1_ruido, W2) + B2
yHat_ruido = sigmoid(Z2_ruido)

print("Saída com ruído:", yHat_ruido)