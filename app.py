from flask import Flask, render_template, request, jsonify
import sys
import os
from inf_acao import get_stock_info, plot_recent_prices
from previsao_fechamento_acao import prepare_data_for_prediction, make_prediction, get_latest_model
import mlflow
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stock_info', methods=['POST'])
def get_stock_information():
    ticker = request.form.get('ticker', 'AMBA')
    
    # Obter informações da ação
    stock_info, dados_recentes = get_stock_info(ticker)
    
    if stock_info is None:
        return jsonify({'error': 'Não foi possível obter informações da ação'})
    
    # Criar gráfico
    plt = plot_recent_prices(dados_recentes, 
                           f"Preços Recentes - {stock_info['Nome Empresa']} ({ticker})")
    
    # Converter gráfico para base64
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return jsonify({
        'stock_info': stock_info,
        'graph': graph_url
    })

@app.route('/make_prediction', methods=['POST'])
def make_stock_prediction():
    try:
        ticker = request.form.get('ticker', 'AMBA')
        sequence_length = int(request.form.get('sequence_length', 60))
        
        # Fazer previsão
        prediction, ultimo_preco, variacao = make_prediction()
        
        if prediction is None:
            return jsonify({'error': 'Erro ao fazer previsão'})
        
        # Capturar o gráfico atual
        img = BytesIO()
        matplotlib.pyplot.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        matplotlib.pyplot.close()
        
        return jsonify({
            'prediction': f"${prediction:.2f}",
            'ultimo_preco': f"${ultimo_preco:.2f}",
            'variacao': f"{variacao:.2f}%",
            'graph': graph_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)