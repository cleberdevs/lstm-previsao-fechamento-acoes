import mlflow
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import pandas as pd

def get_latest_model():
    """Encontrar o modelo mais recente no MLflow"""
    mlflow.set_tracking_uri('file:' + os.path.join(os.getcwd(), 'mlruns'))
    client = mlflow.tracking.MlflowClient()
    
    experiments = client.search_experiments()
    latest_run = None
    latest_timestamp = 0
    
    for experiment in experiments:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs and runs[0].info.start_time > latest_timestamp:
            latest_timestamp = runs[0].info.start_time
            latest_run = runs[0]
    
    if latest_run is None:
        raise Exception("Nenhum modelo encontrado")
        
    return latest_run.info.run_id

def make_prediction():
    try:
        # Configurações
        ticker = 'AMBA'
        sequence_length = 60
        num_sequences = 3  # Usando 3 sequências
        
        # Baixar dados históricos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=sequence_length + 200)
        
        print(f"\nBaixando dados históricos de {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
        ticker_obj = yf.Ticker(ticker)
        dados = ticker_obj.history(start=start_date, end=end_date)
        
        # Carregar modelo
        run_id = get_latest_model()
        print(f"\nCarregando modelo do run_id: {run_id}")
        model = mlflow.keras.load_model(f"runs:/{run_id}/modelo_lstm")
        
        predictions = []
        # Criar as 3 sequências (mesma lógica do código de teste)
        for i in range(num_sequences):
            start_idx = -(sequence_length + i*5)
            end_idx = -i*5 if i > 0 else None
            
            sequence_dates = dados.index[start_idx:end_idx]
            print(f"\nSequência {i+1}:")
            print(f"Início: {sequence_dates[0].strftime('%Y-%m-%d')}")
            print(f"Fim: {sequence_dates[-1].strftime('%Y-%m-%d')}")
            
            close_prices = dados['Close'].values[start_idx:end_idx].reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            X = scaled_data.reshape(1, sequence_length, 1)
            
            # Fazer previsão
            pred_scaled = model.predict(X, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)[0][0]
            predictions.append(pred)
            print(f"Previsão: ${pred:.2f}")
        
        # Calcular previsão final e estatísticas
        prediction = np.mean(predictions)
        prediction_std = np.std(predictions)
        ultimo_preco = float(dados['Close'].iloc[-1])
        variacao = ((prediction - ultimo_preco) / ultimo_preco) * 100
        
        # Plotar gráfico
        plt.figure(figsize=(15, 7))
        
        # Plotar histórico recente
        plt.plot(dados.index[-30:], dados['Close'][-30:], 
                label='Histórico Recente', color='blue', linewidth=2)
        
        # Plotar previsão final
        proxima_data = dados.index[-1] + timedelta(days=1)
        plt.scatter(proxima_data, prediction, 
                   color='red', s=150, label='Previsão Final')
        
        plt.title(f'Previsão de Preço para {ticker}', fontsize=16)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Preço ($)', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        # Adicionar informações
        info_text = (f'Último preço: ${ultimo_preco:.2f}\n'
                    f'Previsão final: ${prediction:.2f}\n'
                    f'Desvio padrão: ${prediction_std:.2f}\n'
                    f'Variação: {variacao:.2f}%\n\n'
                    f'Previsões por sequência:\n')
        for i, pred in enumerate(predictions):
            info_text += f'Seq {i+1}: ${pred:.2f}\n'
        
        plt.figtext(0.01, 0.01, info_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('previsao_atual.png')
        
        # Imprimir resultados
        print("\nResultados Finais:")
        print(f"Data da previsão: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Último preço conhecido: ${ultimo_preco:.2f}")
        print(f"Previsão final (média): ${prediction:.2f}")
        print(f"Desvio padrão: ${prediction_std:.2f}")
        print(f"Variação esperada: {variacao:.2f}%")
        
        # Mostrar gráfico
        plt.show()
        
        return prediction, ultimo_preco, variacao
        
    except Exception as e:
        print(f"Erro ao fazer previsão: {e}")
        return None, None, None

def save_prediction_results(prediction, ultimo_preco, variacao):
    """Salvar resultados da previsão"""
    try:
        data_atual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        resultados = {
            'Data': [data_atual],
            'Último Preço': [f"${ultimo_preco:.2f}"],
            'Previsão': [f"${prediction:.2f}"],
            'Variação (%)': [f"{variacao:.2f}%"]
        }
        
        df = pd.DataFrame(resultados)
        df.to_csv('historico_previsoes.csv', 
                 mode='a', 
                 header=not os.path.exists('historico_previsoes.csv'),
                 index=False)
        
        print("\nResultados salvos em 'historico_previsoes.csv'")
        
    except Exception as e:
        print(f"Erro ao salvar resultados: {e}")

if __name__ == "__main__":
    try:
        prediction, ultimo_preco, variacao = make_prediction()
        
        if all(v is not None for v in [prediction, ultimo_preco, variacao]):
            save_prediction_results(prediction, ultimo_preco, variacao)
            
            if os.path.exists('historico_previsoes.csv'):
                historico = pd.read_csv('historico_previsoes.csv')
                print("\nHistórico de Previsões:")
                print(historico.to_string(index=False))
                
    except Exception as e:
        print(f"Erro na execução: {e}")