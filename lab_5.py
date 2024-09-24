import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def calcular_media_desvio(resultados):
    """Calcula a média e o desvio padrão dos resultados."""
    return np.mean(resultados), np.std(resultados)

def carregar_dados(arquivo='teste2.npy'):
    """Carrega os dados de um arquivo .npy."""
    dados = np.load(arquivo)
    return dados[0], np.ravel(dados[1])

def treinar_avaliar_mlp(x, y):
    """Treina e avalia um regressor MLP."""
    regr = MLPRegressor(
        hidden_layer_sizes=(10,),
        max_iter=1000,
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        n_iter_no_change=50
    )
    
    regr.fit(x, y)  # Treinamento
    y_pred = regr.predict(x)  # Predição
    
    erro = np.mean((y_pred - y) ** 2)  # Cálculo do erro
    plotar_resultados(x, y, regr.loss_curve_, y_pred)  # Plotagem
    
    return erro

def plotar_resultados(x, y, loss_curve, y_pred):
    """Gera gráficos dos resultados."""
    plt.figure(figsize=[14, 7])
    
    plt.subplot(1, 3, 1)
    plt.title('Dados Originais')
    plt.plot(x, y, label='Original')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.title('Curva de Perda')
    plt.plot(loss_curve)
    plt.xlabel('Iterações')
    plt.ylabel('Perda')
    
    plt.subplot(1, 3, 3)
    plt.title('Comparação entre Predição e Dados Originais')
    plt.plot(x, y, linewidth=1, color='red', label='Original')
    plt.plot(x, y_pred, linewidth=2, label='Predito')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Função principal para executar o treinamento e avaliação."""
    resultados_erro = []
    
    for _ in range(10):
        print('Carregando dados...')
        x, y = carregar_dados()
        
        print('Treinando e avaliando o regressor MLP...')
        erro = treinar_avaliar_mlp(x, y)
        resultados_erro.append(erro)
    
    # Cálculo da média e desvio padrão dos erros
    media_erro, desvio_padrao_erro = calcular_media_desvio(resultados_erro)
    print("Média do erro:", media_erro)
    print("Desvio padrão do erro:", desvio_padrao_erro)

if __name__ == "__main__":
    main()
