import numpy as np # biblioteca 

# Definindo os padrões de entrada, onde cada dígito é representado por uma matriz 4x4.
# Estas matrizes são achatadas para se tornarem vetores de 16 elementos (16 características).
X = np.array([
    [0, 1, 1, 0,  # Dígito 0
     1, 0, 0, 1,
     1, 0, 0, 1,
     0, 1, 1, 0],

    [0, 0, 1, 0,  # Dígito 1
     0, 1, 1, 0,
     0, 0, 1, 0,
     0, 0, 1, 0],

    [0, 1, 1, 0,  # Outro exemplo de Dígito 0
     1, 0, 0, 1,
     1, 0, 0, 1,
     0, 1, 1, 0],

    [0, 0, 1, 0,  # Outro exemplo de Dígito 1
     0, 1, 1, 0,
     0, 0, 1, 0,
     0, 0, 1, 0]
])

# Vetor de saída esperada (targets), onde 0 representa o dígito '0' e 1 representa o dígito '1'.
y = np.array([0, 1, 0, 1])

# Inicialização aleatória das sinapses e do bias. As sinapses são inicializadas com valores aleatórios
# subtraindo 0.5 para obter uma distribuição em torno de zero. Bias é inicializado com 1.
sinapse = np.random.rand(16) - 0.5
bias = 1

# Definindo a taxa de aprendizado e o número máximo de épocas para o treinamento (Se a taxa de aprendizado for muito alta, o modelo pode "pular" a solução ótima e não convergir corretamente.).
taxa_aprendizado = 0.1
epocas = 1000

# Função de ativação do tipo degrau, que converte valores negativos para 0 e não negativos para 1.
def ativacao(x):
    return 1 if x >= 0 else 0

# Loop de treinamento do Perceptron por um número máximo de épocas.
for epoca in range(epocas):
    erro_total = 0
    for i in range(len(X)):
        # Cálculo do valor de saída do perceptron para cada exemplo de treinamento.
        soma = np.dot(X[i], sinapse) + bias
        y_pred = ativacao(soma)
        # Cálculo do erro como a diferença entre a saída esperada e a predita.
        erro = y[i] - y_pred
        # Ajuste das sinapses com base no erro calculado, multiplicado pela taxa de aprendizado.
        sinapse += taxa_aprendizado * erro * X[i]
       # bias += taxa_aprendizado * erro
        erro_total += abs(erro)
    
    # Se o erro total for 0, o treinamento é interrompido, indicando convergência.
    if erro_total == 0:
        print(f"Treinamento concluído na época {epoca + 1}")
        break

# Função para testar novos exemplos com o modelo treinado.
def testar(matriz):
    # A matriz de entrada é achatada para um vetor antes de ser processada.
    vetor = np.array(matriz).flatten()
    resultado = np.dot(vetor, sinapse) + bias
    return ativacao(resultado)

# Testando o perceptron com novos exemplos de dígitos '0' e '1'.
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
teste_now = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [1, 1, 0, 1]
]

# Impressão dos resultados para verificar a corretude do modelo treinado.
print("Resultado para o dígito 0:", testar(teste_0))  # Esperado: 0
print("Resultado para o dígito 1:", testar(teste_1))  # Esperado: 1
print("Resultado da matriz testada:", testar(teste_now))  # Esperado: 1