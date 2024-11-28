```markdown
# Previsão de Preços de Ações usando LSTM

Um aplicativo web que utiliza redes neurais LSTM (Long Short-Term Memory) para prever preços de fechamento de ações. O projeto foi desenvolvido especificamente para análise e previsão das ações da Ambarella Inc. (AMBA).

## Funcionalidades

- Consulta de informações detalhadas de ações via Yahoo Finance
- Visualização de gráficos de preços históricos
- Previsão de preços futuros usando modelo LSTM
- Interface web intuitiva para interação com o sistema
- Registro de previsões e métricas de desempenho
- Comparação de períodos históricos diferentes

## Tecnologias Utilizadas

- Python 3.11
- TensorFlow/Keras para modelagem LSTM
- MLflow para gerenciamento de experimentos
- Flask para servidor web
- yfinance para dados de mercado
- Pandas e NumPy para manipulação de dados
- Matplotlib para visualizações
- Bootstrap e jQuery para frontend

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/lstm-previsao-fechamento-acoes.git
cd lstm-previsao-fechamento-acoes
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o treinamento do modelo:
```bash
python criacao_modelo.py
```

4. Inicie o servidor web:
```bash
python app.py
```

5. Acesse a aplicação em `http://localhost:5000`

## Estrutura do Projeto

- `app.py`: Servidor Flask e endpoints da API
- `criacao_modelo.py`: Script para treinar o modelo LSTM
- `previsao_fechamento_acao.py`: Lógica de previsão usando o modelo treinado
- `inf_acao.py`: Funções para obter informações das ações
- `comparacao_periodos.py`: Análise comparativa de diferentes períodos históricos
- `templates/`: Arquivos HTML da interface web

## Uso

1. Acesse a interface web
2. Digite o ticker da ação (AMBA) para visualizar informações
3. Configure o período de análise desejado
4. Clique em "Fazer Previsão" para obter a previsão de preço

## Limitações

- O modelo atual foi treinado especificamente para ações da AMBA
- Previsões são baseadas apenas em preços históricos
- Desempenho pode variar em períodos de alta volatilidade

## Métricas e Avaliação

O modelo é avaliado usando:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- Comparação visual entre valores previstos e reais

