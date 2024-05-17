import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st

# Função para carregar e fazer previsões para um modelo específico
def carregar_e_prever_modelo(modelo_arquivo, x_test):
    modelo = joblib.load(modelo_arquivo)
    return modelo.predict(x_test)

# Carregar o DataFrame
df = pd.read_csv('C:\\Users\\João Pedro\\Desktop\\brent-price\\data\\ipea.csv')
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)

# Recursos de Atraso (lag features) para séries temporais
lags = 7
for lag in range(1, lags + 1):
    df[f'Preço_lag_{lag}'] = df['Preço - petróleo bruto - Brent (FOB)'].shift(lag)

# Remove linhas com valores Nulos
df = df.dropna()

# Preparando os dados para treinamento
x = df[['Preço_lag_1', 'Preço_lag_2', 'Preço_lag_3', 'Preço_lag_4']].values  # Inputs são os preços atrasados
y = df['Preço - petróleo bruto - Brent (FOB)'].values  # Output é o preço atual

# Dividir os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

# Modelos disponíveis
modelos = {
    "GradientBoostingRegressor": "modelo_brent_GB.pkl",
    "RandomForestRegressor": "modelo_brent_RF.pkl",
    "AdaBoostRegressor": "modelo_brent_AB.pkl",
    "CatBoostRegressor": "modelo_brent_CB.pkl",
    "LGBMRegressor": "modelo_brent_LG.pkl",
    "XGBRegressor": "modelo_brent_XG.pkl"
}

# Descrições dos modelos
descricoes_modelos = {
    "GradientBoostingRegressor": "Utiliza uma técnica que combina várias previsões simples para criar uma previsão mais precisa. Imagine como se várias opiniões fossem reunidas para chegar a uma conclusão melhor.",
    "RandomForestRegressor": "Cria várias 'árvores de decisão' independentes e combina suas previsões para melhorar a precisão. Pense nisso como consultar vários especialistas e combinar suas opiniões.",
    "AdaBoostRegressor": "Foca mais nos erros anteriores para melhorar a previsão. É como um treinador que ajusta suas estratégias com base nos erros passados para obter melhores resultados.",
    "CatBoostRegressor": "É bom em lidar com dados complexos e ruidosos, como informações categóricas. Funciona bem com diferentes tipos de dados e é resiliente a informações confusas.",
    "LGBMRegressor": "Usa uma abordagem eficiente em termos de memória e velocidade para processar grandes volumes de dados. É como um computador rápido e econômico que ainda assim oferece ótimos resultados.",
    "XGBRegressor": "É uma versão otimizada de técnicas de previsão avançadas, conhecida por sua alta performance. Pense nisso como um carro esportivo ajustado para máxima eficiência e velocidade."
}

# Escolha do modelo para o primeiro gráfico
modelo_selecionado_1 = st.sidebar.selectbox("Selecione o modelo para o primeiro gráfico", list(modelos.keys()))

st.markdown(f"### Previsões do Preço do Petróleo Brent - {modelo_selecionado_1}")
st.markdown(descricoes_modelos[modelo_selecionado_1])

# Carregar o modelo treinado e fazer previsões para o primeiro gráfico
with st.spinner(f"Carregando e fazendo previsões com {modelo_selecionado_1}..."):
    arquivo_modelo_1 = modelos[modelo_selecionado_1]
    modelo_brent_1 = joblib.load(arquivo_modelo_1)
    predictions_1 = modelo_brent_1.predict(x_test)
    mse_1 = mean_squared_error(y_test, predictions_1)
    mae_1 = mean_absolute_error(y_test, predictions_1)

    # Fazer previsões para a próxima semana usando os últimos dados conhecidos
    last_known_data = x[-1].reshape(1, -1)
    next_week_predictions_1 = []
    for _ in range(7):  # para cada dia da próxima semana
        next_day_pred = modelo_brent_1.predict(last_known_data)[0]
        next_week_predictions_1.append(next_day_pred)
        last_known_data = np.roll(last_known_data, -1)
        last_known_data[0, -1] = next_day_pred

    # As datas correspondentes à próxima semana
    next_week_dates_1 = pd.date_range(df['Data'].iloc[-1], periods=8)[1:]

    # Selecionar os dados da semana atual
    current_week_dates_1 = df['Data'].iloc[-7:]
    current_week_prices_1 = df['Preço - petróleo bruto - Brent (FOB)'].iloc[-7:]

    # Plotar os preços reais da semana atual e as previsões para a próxima semana (Gráfico 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(current_week_dates_1, current_week_prices_1, 'bo-', label='Preços Atuais')
    ax.plot(next_week_dates_1, next_week_predictions_1, 'r--o', label='Previsões para a Próxima Semana')

    # Adicionar valores reais e previstos acima dos pontos do gráfico
    for i, txt in enumerate(current_week_prices_1):
        ax.annotate(f'{txt:.1f}', (current_week_dates_1.iloc[i], current_week_prices_1.iloc[i] + 0.15), ha='center', color='blue')
    for i, txt in enumerate(next_week_predictions_1):
        ax.annotate(f'{txt:.1f}', (next_week_dates_1[i], next_week_predictions_1[i] + 0.15), ha='center', color='red')

    # Formatando o eixo x para apresentar as datas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))  # Formatar datas como 'Ano-Mes-Dia'
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Escolher automaticamente a localização das datas

    # Melhorar a leitura girando as datas e ajustando o espaçamento
    fig.autofmt_xdate()  # Gira as datas para evitar sobreposição

    # Adicionar legendas e título
    ax.legend()
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    ax.grid(True)
    ax.set_title('Preços Reais e Previsões de Preços para a Próxima Semana')

    # Adicionar as métricas de erro no topo do gráfico, centralizadas
    fig.text(0.35, 0.80, f'Mean Squared Error: {mse_1:.2f}', ha='center', va='center', fontsize=12)
    fig.text(0.35, 0.85, f'Mean Absolute Error: {mae_1:.2f}', ha='center', va='center', fontsize=12)

    st.pyplot(fig)

# Escolha do modelo para o segundo gráfico
modelo_selecionado_2 = st.sidebar.selectbox("Selecione o modelo para o segundo gráfico", list(modelos.keys()))

st.markdown(f"### Previsões do Preço do Petróleo Brent - {modelo_selecionado_2}")
st.markdown(descricoes_modelos[modelo_selecionado_2])

# Carregar o modelo treinado e fazer previsões para o segundo gráfico
with st.spinner(f"Carregando e fazendo previsões com {modelo_selecionado_2}..."):
    arquivo_modelo_2 = modelos[modelo_selecionado_2]
    modelo_brent_2 = joblib.load(arquivo_modelo_2)
    predictions_2 = modelo_brent_2.predict(x_test)
    mse_2 = mean_squared_error(y_test, predictions_2)
    mae_2 = mean_absolute_error(y_test, predictions_2)

    # Fazer previsões para a próxima semana usando os últimos dados conhecidos
    last_known_data = x[-1].reshape(1, -1)
    next_week_predictions_2 = []
    for _ in range(7):  # para cada dia da próxima semana
        next_day_pred = modelo_brent_2.predict(last_known_data)[0]
        next_week_predictions_2.append(next_day_pred)
        last_known_data = np.roll(last_known_data, -1)
        last_known_data[0, -1] = next_day_pred

    # As datas correspondentes à próxima semana
    next_week_dates_2 = pd.date_range(df['Data'].iloc[-1], periods=8)[1:]

    # Selecionar os dados da semana atual
    current_week_dates_2 = df['Data'].iloc[-7:]
    current_week_prices_2 = df['Preço - petróleo bruto - Brent (FOB)'].iloc[-7:]

    # Plotar os preços reais da semana atual e as previsões para a próxima semana (Gráfico 2)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(current_week_dates_2, current_week_prices_2, 'bo-', label='Preços Atuais')
    ax2.plot(next_week_dates_2, next_week_predictions_2, 'r--o', label='Previsões para a Próxima Semana')

    # Adicionar valores reais e previstos acima dos pontos do gráfico
    for i, txt in enumerate(current_week_prices_2):
        ax2.annotate(f'{txt:.1f}', (current_week_dates_2.iloc[i], current_week_prices_2.iloc[i] + 0.15), ha='center', color='blue')
    for i, txt in enumerate(next_week_predictions_2):
        ax2.annotate(f'{txt:.1f}', (next_week_dates_2[i], next_week_predictions_2[i] + 0.15), ha='center', color='red')

    # Formatando o eixo x para apresentar as datas
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))  # Formatar datas como 'Ano-Mes-Dia'
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())  # Escolher automaticamente a localização das datas

    # Melhorar a legibilidade girando as datas e ajustando o espaçamento
    fig2.autofmt_xdate()  # Gira as datas para evitar sobreposição

    # Adicionar legendas e título
    ax2.legend()
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Preço')
    ax2.grid(True)
    ax2.set_title('Preços Reais e Previsões de Preços para a Próxima Semana')

    # Adicionar as métricas de erro no topo do gráfico, centralizadas
    fig2.text(0.35, 0.80, f'Mean Squared Error: {mse_2:.2f}', ha='center', va='center', fontsize=12)
    fig2.text(0.35, 0.85, f'Mean Absolute Error: {mae_2:.2f}', ha='center', va='center', fontsize=12)

    st.pyplot(fig2)

# Comparar os modelos
st.markdown(f"### Comparação de Modelos", unsafe_allow_html=True)
comparison_html = f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <div style="margin-right: 50px; text-align: center;">
            <p>{modelo_selecionado_1} vs {modelo_selecionado_2}</p>
            <p>{'Modelo 1' if mse_1 < mse_2 else 'Modelo 2'} tem um MSE menor ({mse_1:.2f} vs {mse_2:.2f})</p>
            <p>{'Modelo 1' if mae_1 < mae_2 else 'Modelo 2'} tem um MAE menor ({mae_1:.2f} vs {mae_2:.2f})</p>
        </div>
    </div>
"""
st.markdown(comparison_html, unsafe_allow_html=True)
