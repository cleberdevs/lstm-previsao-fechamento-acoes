import os
import numpy as np
import mlflow
import mlflow.keras
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

def test_multiple_sequences(ticker='AMBA', sequence_length=60, max_sequences=20):
    """Testar diferentes números de sequências e avaliar a precisão"""
    try:
        print(f"\nIniciando teste de sequências para {ticker}")
        print(f"Testando de 1 até {max_sequences} sequências")
        
        # Baixar dados históricos extras para teste
        end_date = datetime.now()
        start_date = end_date - timedelta(days=sequence_length + 200)
        
        print(f"\nBaixando dados históricos de {start_date.strftime('%Y-%m-%d')} até {end_date.strftime('%Y-%m-%d')}")
        ticker_obj = yf.Ticker(ticker)
        dados = ticker_obj.history(start=start_date, end=end_date)
        
        print(f"Total de dias baixados: {len(dados)}")
        
        # Carregar modelo
        run_id = get_latest_model()
        print(f"\nCarregando modelo do run_id: {run_id}")
        model = mlflow.keras.load_model(f"runs:/{run_id}/modelo_lstm")
        
        resultados = []
        todas_previsoes = []
        
        print("\nRealizando previsões com diferentes números de sequências...")
        # Testar diferentes números de sequências
        for num_sequences in range(1, max_sequences + 1):
            print(f"Testando com {num_sequences} sequência(s)...")
            predictions = []
            
            # Criar múltiplas sequências
            for i in range(num_sequences):
                start_idx = -(sequence_length + i*5)
                end_idx = -i*5 if i > 0 else None
                
                close_prices = dados['Close'].values[start_idx:end_idx].reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(close_prices)
                X = scaled_data.reshape(1, sequence_length, 1)
                
                # Fazer previsão
                pred_scaled = model.predict(X, verbose=0)
                pred = scaler.inverse_transform(pred_scaled)[0][0]
                predictions.append(pred)
            
            # Calcular estatísticas
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            resultados.append({
                'num_sequences': num_sequences,
                'mean_prediction': mean_pred,
                'std_prediction': std_pred,
                'min_prediction': min(predictions),
                'max_prediction': max(predictions),
                'range': max(predictions) - min(predictions),
                'coefficient_variation': (std_pred/mean_pred)*100 if mean_pred != 0 else 0
            })
            
            todas_previsoes.append(predictions)
        
        # Criar visualizações
        print("\nGerando visualizações...")
        
        # Figura 1: Análise Estatística
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Previsão Média e Desvio
        plt.subplot(2, 1, 1)
        means = [r['mean_prediction'] for r in resultados]
        stds = [r['std_prediction'] for r in resultados]
        nums = [r['num_sequences'] for r in resultados]
        
        plt.errorbar(nums, means, yerr=stds, fmt='o-', capsize=5, color='blue')
        plt.fill_between(nums, 
                        [m-s for m,s in zip(means,stds)],
                        [m+s for m,s in zip(means,stds)],
                        alpha=0.2, color='blue')
        plt.title('Previsão Média e Desvio Padrão por Número de Sequências', fontsize=12)
        plt.xlabel('Número de Sequências')
        plt.ylabel('Preço Previsto ($)')
        plt.grid(True)
        
        # Subplot 2: Coeficiente de Variação
        plt.subplot(2, 1, 2)
        cv = [r['coefficient_variation'] for r in resultados]
        plt.plot(nums, cv, 'r.-')
        plt.title('Coeficiente de Variação por Número de Sequências', fontsize=12)
        plt.xlabel('Número de Sequências')
        plt.ylabel('Coeficiente de Variação (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('analise_estatistica_sequencias.png')
        
        # Figura 2: Distribuição das Previsões
        plt.figure(figsize=(15, 8))
        plt.boxplot(todas_previsoes, labels=nums)
        plt.title('Distribuição das Previsões por Número de Sequências', fontsize=12)
        plt.xlabel('Número de Sequências')
        plt.ylabel('Preço Previsto ($)')
        plt.grid(True)
        plt.savefig('distribuicao_previsoes.png')
        
        # Encontrar número ótimo de sequências
        cv_changes = np.diff([r['coefficient_variation'] for r in resultados])
        optimal_sequences = np.where(np.abs(cv_changes) < 0.5)[0][0] + 1
        
        # Salvar resultados em CSV
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv('resultados_sequencias.csv', index=False)
        
        print("\nAnálise de Sequências:")
        print(f"Número ótimo sugerido de sequências: {optimal_sequences}")
        print("\nResultados detalhados:")
        print(f"{'N.Seq':>5} {'Média($)':>10} {'Std($)':>8} {'CV(%)':>8} {'Range($)':>8}")
        print("-" * 50)
        for r in resultados:
            print(f"{r['num_sequences']:5d} {r['mean_prediction']:10.2f} "
                  f"{r['std_prediction']:8.2f} {r['coefficient_variation']:8.2f} "
                  f"{r['range']:8.2f}")
        
        print("\nArquivos gerados:")
        print("- analise_estatistica_sequencias.png")
        print("- distribuicao_previsoes.png")
        print("- resultados_sequencias.csv")
        
        return optimal_sequences, resultados
        
    except Exception as e:
        print(f"Erro ao testar sequências: {e}")
        return None, None

if __name__ == "__main__":
    max_sequences = int(input("Digite o número máximo de sequências para testar (recomendado: 20): "))
    optimal_sequences, results = test_multiple_sequences(max_sequences=max_sequences)