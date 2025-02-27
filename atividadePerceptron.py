import numpy as np

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

# Inicializar pesos e bias aleatoriamente
pesos = np.random.rand(16) - 0.5
bias = np.random.rand(1) - 0.5
taxa_aprendizado = 0.1
epocas = 1000

# Função de ativação (degrau)
def ativacao(x):
    return 1 if x >= 0 else 0

# Treinamento do Perceptron
for epoca in range(epocas):
    erro_total = 0
    for i in range(len(X)):
        soma = np.dot(X[i], pesos) + bias
        y_pred = ativacao(soma)
        erro = y[i] - y_pred
        pesos += taxa_aprendizado * erro * X[i]
        bias += taxa_aprendizado * erro
        erro_total += abs(erro)
    
    if erro_total == 0:
        print(f"Treinamento concluído na época {epoca + 1}")
        break

# Teste com um novo exemplo
def testar(matriz):
    vetor = np.array(matriz).flatten()
    resultado = np.dot(vetor, pesos) + bias
    return ativacao(resultado)

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