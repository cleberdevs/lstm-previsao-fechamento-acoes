# lstm.py
import os
import numpy as np
import mlflow
import mlflow.keras
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Definindo a URI do MLflow
#os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.49.2:30500"  # Ajuste conforme necessário
print("MLflow URI configurada.")

# Iniciando uma execução do MLflow e armazenando o run_id
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"O run_id é: {run_id}")

    # Definindo o ticker da empresa
    ticker = 'AMBA'
    print(f"Baixando dados históricos para o ticker: {ticker}")

    # Obtendo os dados históricos dos últimos 5 anos
    dados_historicos = yf.download(ticker, start='2018-01-01', end='2023-01-01')
    print("Dados históricos baixados com sucesso.")

    # Processando os dados
    data = dados_historicos['Close'].values.reshape(-1, 1)  # Usando os dados de fechamento
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print("Dados escalonados.")

    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len - 60:, :]

    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print("Dados de treinamento preparados.")

    # Criando o modelo LSTM
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))  # Usando Input como a primeira camada
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Modelo LSTM criado e compilado.")

    # Treinando o modelo
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    print("Modelo treinado com sucesso.")

    # Definindo um exemplo de entrada
    input_example = np.array([[0.1, 0.2, 0.3]])  # Ajuste conforme a forma de entrada do seu modelo

    # Inferir a assinatura do modelo
    signature = ModelSignature(
        inputs=Schema([ColSpec("float", "input_1"), ColSpec("float", "input_2"), ColSpec("float", "input_3")]),
        outputs=Schema([ColSpec("float", "output")])  # Ajuste conforme necessário
    )

    # Registrar o modelo no MLflow com assinatura e exemplo de entrada
    mlflow.keras.log_model(
        model,
        "modelo_lstm",
        signature=signature,
        input_example=input_example
    )
    print("Modelo registrado no MLflow.")

    # Adicionando o modelo ao registro de modelos
    mlflow.register_model(f"runs:/{run_id}/modelo_lstm", "NomeDoModelo")
    print("Modelo adicionado ao registro de modelos.")

# Finalizando a execução do MLflow
mlflow.end_run()
print("Execução do MLflow finalizada.")


