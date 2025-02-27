import numpy as np
from sklearn.linear_model import Perceptron

# Definir os padrões de entrada (achatando as matrizes 4x4 para vetores de 16 elementos)
X = np.array([
    [0, 1, 1, 0,  # Digito 0
     1, 0, 0, 1,
     1, 0, 0, 1,
     0, 1, 1, 0],

    [0, 0, 1, 0,  # Digito 1
     0, 1, 1, 0,
     0, 0, 1, 0,
     0, 0, 1, 0],

    [0, 1, 1, 0,  # Outro Digito 0
     1, 0, 0, 1,
     1, 0, 0, 1,
     0, 1, 1, 0],

    [0, 0, 1, 0,  # Outro Digito 1
     0, 1, 1, 0,
     0, 0, 1, 0,
     0, 0, 1, 0]
])

# Saída esperada (1 para "1", 0 para "0")
y = np.array([0, 1, 0, 1])

# Criando e treinando o Perceptron
modelo = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
modelo.fit(X, y)

# Teste com novos exemplos
def testar(matriz):
    vetor = np.array(matriz).flatten()
    return modelo.predict([vetor])[0]

# Exemplo de teste
teste_0 = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]

teste_1 = [
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 0]
]

print("Resultado para o dígito 0:", testar(teste_0))  # Esperado: 0
print("Resultado para o dígito 1:", testar(teste_1))  # Esperado: 1