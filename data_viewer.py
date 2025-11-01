# data_viewer.py
import sqlite3
import pandas as pd
import json
import re
import os
from flask import Flask, jsonify, send_from_directory, Response
import webbrowser
from threading import Timer

# --- Configuração ---
# Pega as configurações do seu main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets/"
DATABASE_PATH = os.path.join(BASE_DIR, "logs_db.db")
# --------------------

app = Flask(__name__)

def get_db_connection():
    """Conecta ao banco de dados"""
    return sqlite3.connect(DATABASE_PATH)

def get_valid_table_names() -> list:
    """Helper de segurança: Pega uma lista de todas as tabelas válidas no DB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # Transforma a lista de tuplas [('tabela1',), ('tabela2',)] em ['tabela1', 'tabela2']
    valid_names = [item[0] for item in cursor.fetchall()]
    conn.close()
    return valid_names

# --- API Endpoints ---

@app.route('/api/logs')
def get_logs():
    """
    Endpoint 1: Encontra todos os 'log_id' (sufixos de tabela)
    para preencher o primeiro dropdown.
    """
    valid_tables = get_valid_table_names()
    log_ids = set()
    
    for table_name in valid_tables:
        # Procura pelo primeiro underscore (_) seguido por um dígito (\d)
        # (ex: ..._2025...)
        match = re.search(r'(_\d.*)', table_name)
        
        if match:
            # Pega o grupo capturado (ex: '_2025_09_03...')
            # e remove o primeiro underscore
            log_id = match.group(1)[1:]
            log_ids.add(log_id)
            
    return jsonify(list(log_ids))

@app.route('/api/tables/<log_id>')
def get_tables_for_log(log_id):
    """
    Endpoint 2: Dado um log_id, encontra todas as tabelas
    associadas a ele (ex: 'GPS_log_id', 'dados_variados_log_id').
    """
    valid_tables = get_valid_table_names()
    
    # Filtra a lista para incluir apenas tabelas que terminam com o log_id
    tables_for_log = [name for name in valid_tables if name.endswith(f"_{log_id}")]
    
    return jsonify(tables_for_log)

@app.route('/api/data/<table_name>')
def get_table_data(table_name):
    """
    Endpoint 3: Pega todos os dados de uma tabela específica e
    retorna como JSON.
    """
    
    # --- Medida de Segurança (Previne SQL Injection) ---
    # Verifica se o nome da tabela solicitado está na nossa lista segura.
    valid_tables = get_valid_table_names()
    if table_name not in valid_tables:
        return jsonify({"error": "Nome de tabela inválido"}), 400
    
    # Se for seguro, executa a consulta
    conn = get_db_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        # Converte o DataFrame para JSON no formato 'records'
        # (uma lista de dicionários, ex: [{"col1": 1, "col2": "a"}])
        return df.to_json(orient='records')
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500


@app.route('/api/export/<table_name>')
def export_table_csv(table_name):
    """
    Endpoint 4: Pega todos os dados de uma tabela específica
    e retorna como um arquivo .csv para download.
    """
    
    # --- Medida de Segurança (Previne SQL Injection) ---
    valid_tables = get_valid_table_names()
    if table_name not in valid_tables:
        return jsonify({"error": "Nome de tabela inválido"}), 400
    
    # Se for seguro, executa a consulta
    conn = get_db_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        # Converte o DataFrame para um CSV em memória
        csv_data = df.to_csv(index=False, encoding='utf-8')
        
        # Retorna o CSV como um download
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition":
                     f"attachment; filename={table_name}.csv"}
        )
    
    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500


# --- Servindo o Front-End ---

@app.route('/')
def serve_index():
    """Endpoint 4: Serve a página HTML principal (index.html)."""
    # Diz ao Flask para enviar o arquivo 'index.html' do diretório atual ('.')
    return send_from_directory('.', 'index.html')
# --- FIM DA ADIÇÃO ---

@app.route('/app.js')
def serve_js():
    """Endpoint 5: Serve o arquivo JavaScript."""
    return send_from_directory('.', 'app.js')

# --- BLOCO MODIFICADO (agora sem modificação, apenas o 'main') ---

def open_browser():
    """Função para abrir o navegador."""
    # Abre a URL no seu navegador padrão
    webbrowser.open_new_tab("http://127.0.0.1:5000/")

if __name__ == '__main__':
    print("Iniciando Data Viewer em http://127.0.0.1:5000")
    print("O navegador será aberto automaticamente...")
    
    # Cria um "Timer" para chamar a função open_browser() 
    # 1 segundo *após* o servidor começar.
    # Isso dá tempo ao Flask para iniciar antes de tentar abrir a página.
    Timer(1, open_browser).start()
    
    # Inicia o servidor Flask
    # (Definimos use_reloader=False para evitar que o navegador abra duas vezes no modo debug)
    app.run(debug=True, port=5000, use_reloader=False)