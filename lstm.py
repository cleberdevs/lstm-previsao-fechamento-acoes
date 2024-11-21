import os
import numpy as np
import mlflow
import mlflow.keras
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# Configurar MLflow
mlflow.set_tracking_uri('file:' + os.path.join(os.getcwd(), 'mlruns'))

# Configurar ambiente conda
conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': [
        'python=3.11',
        'tensorflow',
        'numpy',
        'pandas',
        'scikit-learn',
        'yfinance',
        'mlflow',
        'matplotlib'
    ],
    'name': 'lstm_env'
}

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"O run_id é: {run_id}")

    # Baixar dados
    ticker = 'AMBA'
    print(f"Baixando dados históricos para o ticker: {ticker}")
    dados_historicos = yf.download(ticker, start='2019-01-01', end= datetime.now().strftime('%Y-%m-%d'))
    
    # Processar dados
    data = dados_historicos['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Preparar dados de treinamento e teste
    training_data_len = int(np.ceil(len(scaled_data) * 0.8))
    train_data = scaled_data[0:training_data_len, :]
    test_data = scaled_data[training_data_len:, :]

    # Função para criar dataset
    def create_dataset(data, time_steps=60):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps, 0])
        return np.array(X), np.array(y)

    # Criar datasets de treino e teste
    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    # Reshape para o formato [amostras, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Criar e treinar modelo
    model = Sequential([
        Input(shape=(60, 1), name='input_1'),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, batch_size=1, epochs=1, verbose=1)

    # Fazer previsões
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverter normalização
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform([y_test])

    # Criar gráfico
    plt.figure(figsize=(20,10))

    # Plotar dados de treino
    train_dates = dados_historicos.index[60:training_data_len]
    plt.plot(train_dates, y_train_inv.T, 'b', label='Treino - Real')
    plt.plot(train_dates, train_predict, 'r--', label='Treino - Previsto')

    # Plotar dados de teste
    test_dates = dados_historicos.index[training_data_len+60:len(dados_historicos)]
    plt.plot(test_dates, y_test_inv.T, 'g', label='Teste - Real')
    plt.plot(test_dates, test_predict, 'orange', label='Teste - Previsto')

    plt.title(f'Previsão vs Valor Real - {ticker} (2019-2024)', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Preço ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    
    # Calcular métricas
    train_mae = mean_absolute_error(y_train_inv.T, train_predict)
    train_rmse = np.sqrt(mean_squared_error(y_train_inv.T, train_predict))
    test_mae = mean_absolute_error(y_test_inv.T, test_predict)
    test_rmse = np.sqrt(mean_squared_error(y_test_inv.T, test_predict))

    # Adicionar métricas ao gráfico
    metrics_text = f'Métricas de Treino:\nMAE: {train_mae:.2f}\nRMSE: {train_rmse:.2f}\n\n'
    metrics_text += f'Métricas de Teste:\nMAE: {test_mae:.2f}\nRMSE: {test_rmse:.2f}'
    
    plt.figtext(0.01, 0.01, metrics_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()

    # Salvar o gráfico
    plt.savefig('previsoes_completas.png', dpi=300, bbox_inches='tight')
    
    # Log do gráfico e métricas no MLflow
    mlflow.log_artifact('previsoes_completas.png')
    mlflow.log_metrics({
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse
    })

    # Definir assinatura do modelo
    signature = ModelSignature(
        inputs=Schema([
            TensorSpec(np.dtype('float32'), (-1, 60, 1), name='input_1')
        ]),
        outputs=Schema([
            TensorSpec(np.dtype('float32'), (-1, 1), name='output')
        ])
    )

    try:
        # Log parâmetros e modelo
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("epochs", 1)
        mlflow.log_param("batch_size", 1)

        mlflow.keras.log_model(
            model,
            "modelo_lstm",
            signature=signature,
            conda_env=conda_env
        )
        print("Modelo registrado no MLflow.")
        
        # Imprimir métricas
        print(f"\nMétricas de Avaliação:")
        print(f"Treino - MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
        print(f"Teste - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")

    except Exception as e:
        print(f"Erro ao registrar modelo: {e}")
        raise

print("Execução do MLflow finalizada.")

# Mostrar o gráfico
# plt.show()