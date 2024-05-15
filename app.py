from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.dates as mdates


st.markdown("# Modelo preditivo do Petróleo Brent! ")
with open('modelo_brent.pkl', 'rb') as file_2:
    modelo_brent = joblib.load(file_2)


#Carregar o DataFrame
df = pd.read_csv('/mount/src/-brent-price/data/ipea.csv')
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values(by='Data', ascending=True).reset_index(drop=True)

st.dataframe(df)

#Recuros de Atraso (lag features) para séries temporais
lags = 7
for lag in range(1,lags+1): #Criar atrasos de 1 dia até 3 dias
    df[f'Preço_lag_{lag}'] = df['Preço - petróleo bruto - Brent (FOB)'].shift(lag)

# Remove linhas com valores NAN
df = df.dropna()

#Preparando os dados pra terinamento
x = df[['Preço_lag_1','Preço_lag_2','Preço_lag_3','Preço_lag_4']].values #Inputs são os preços atrasados
y = df['Preço - petróleo bruto - Brent (FOB)'].values #Output é o preço atual

#Dividir os dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)


#Fazer previsões
predictions = modelo_brent.predict(x_test)

#Avaliar o modelo
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

st.markdown(f'# O erro quadrado médio é de: {mse:.2f}')
st.markdown(f'# O erro absoluto médio é de: {mae:.2f}')


#Fazer previsões para a próxima semana usando os últimos dados conhecidos
last_known_data = x[-1].reshape(1,-1)
next_week_predictions = []
for _ in range(7): #para cada dia da próxima semana
    next_day_pred = modelo_brent.predict(last_known_data)[0]
    next_week_predictions.append(next_day_pred)
    last_known_data = np.roll(last_known_data, -1)
    last_known_data[0, -1] = next_day_pred

#As datas correspondentes a próxima semana
next_week_dates = pd.date_range(df['Data'].iloc[-1], periods=8)[1:]

#Selecionar os dados da semana atual
current_week_dates = df['Data'].iloc[-7:]
current_week_prices = df['Preço - petróleo bruto - Brent (FOB)'].iloc[-7:]

for week, pred in zip(next_week_dates, next_week_predictions):
    print(f'{week}: {pred:.2f}')

#Plotar os preços reais da semana atual e as previsões para a próxima semana
st.markdown("#Previsões: Uma semana para frente")
plt.figure(figsize=(10,5))
plt.plot(current_week_dates,current_week_prices, 'bo-', label='Preços Atuais')
plt.plot(next_week_dates,next_week_predictions, 'r--o', label='Previsões para a Próxima Semana')

#Formatar o eixo x para apesentar as datas
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))    #Formatar datas como 'Ano-Mes-Dia'
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())   #Escolher automaticamente a localização das datas

#Melhorar a legibilidade irando as datas e ajustando o espaçamento
plt.gcf().autofmt_xdate()    #Gira as datas para evitar sobreposição

plt.legend()
plt.xlabel('Data')
plt.ylabel('Preço')
plt.grid(True)
plt.title('Preços Reais e Previsões dos Preços para as próximas duas semanas')
plt.show()


st.pyplot(plt)

