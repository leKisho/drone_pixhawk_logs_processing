# infrastructure/sql_repository.py
import sqlite3
import pandas as pd
import numpy as np
import os
import re  # <--- ADICIONE ESTA LINHA
from typing import List, Tuple, Dict, Any, Callable

# Importa o "Contrato" que esta classe deve seguir
from scr.domain.interfaces import ILogRepository
# Importa os parsers de log, pois ele ainda precisa extrair o log na primeira vez
from scr.infrastructure.log_parser import LogLabeler, LogSeparator
# Importa o FileRepository para "roubar" a lógica de cache do .npz
from scr.infrastructure.file_repository import FileRepository 

class SQLRepository(ILogRepository):
    """
    Implementação SQL do Repositório.
    Lida com todo o acesso a um banco de dados SQLite.
    
    A lógica é: "Tente ler do SQL. Se falhar, execute a extração
    e salve no SQL para a próxima vez."
    """
    
    def __init__(self, log_filepath: str, db_name: str = "drone_logs.db"):
        self.log_filepath = log_filepath
        self.log_dir = os.path.dirname(log_filepath)
        self.log_filename_no_ext = os.path.basename(log_filepath).split('.')[0]

        # O 'log_id' é usado para nomear as tabelas

        # Substitui qualquer caractere que NÃO seja letra, número ou underscore por um '_'
        self.log_id = re.sub(r'\W+', '_', self.log_filename_no_ext)

        # (O resto do __init__ continua igual)
        self.db_path = os.path.join(self.log_dir, '..', db_name)
        self.conn = sqlite3.connect(self.db_path)
        print(f"Repositório SQL conectado ao banco: {self.db_path}")
        print(f"ID do Log para tabelas SQL: {self.log_id}") # Bom para debug


    def __del__(self):
        # Garante que a conexão com o banco seja fechada
        if self.conn:
            self.conn.close()

    def _table_exists(self, table_name: str) -> bool:
        """Helper para verificar se uma tabela já existe no banco."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        return cursor.fetchone() is not None

    def get_raw_sensor_data(self, required_sensors: Tuple[str, ...]) -> Dict[str, pd.DataFrame]:
        """
        *** A LÓGICA PRINCIPAL ***
        Verifica se os dados já estão no SQL. Se não, parseia o log e os salva.
        """
        sensor_data = {}
        sensores_faltantes = []

        for sensor_name in required_sensors:
            # O nome da tabela é unico por sensor E por log
            table_name = f"{sensor_name}_{self.log_id}"

            if self._table_exists(table_name):
                # CASO 1: JÁ EXISTE NO BANCO (Rápido)
                print(f"Carregando '{sensor_name}' do banco de dados SQL (Tabela: {table_name})...")
                sensor_data[sensor_name] = pd.read_sql(f"SELECT * FROM {table_name}", self.conn)
            else:
                # CASO 2: NÃO EXISTE (Lento)
                print(f"Tabela '{table_name}' não encontrada no SQL.")
                sensores_faltantes.append(sensor_name)
        
        # Se algum sensor faltou, precisamos extrair TODOS do log
        if sensores_faltantes:
            print(f"Extraindo dados do .log para o SQL...")
            self._run_full_log_extraction()
            
            # Tenta carregar novamente agora que os dados devem existir
            for sensor_name in sensores_faltantes:
                table_name = f"{sensor_name}_{self.log_id}"
                if self._table_exists(table_name):
                    print(f"Recarregando '{sensor_name}' do SQL...")
                    sensor_data[sensor_name] = pd.read_sql(f"SELECT * FROM {table_name}", self.conn)
                else:
                    print(f"Aviso: Mesmo após a extração, '{sensor_name}' não foi encontrado.")
                    
        return sensor_data

    def _run_full_log_extraction(self):
        """
        Rotina chamada na primeira vez. Lê o .log e salva CADA sensor
        em sua própria tabela no SQL.
        """
        labeler = LogLabeler(self.log_filepath)
        ids, labels = labeler.extract_labels()
        
        print(f"Encontrados {len(labels)} tipos de dados. Salvando no SQL...")
        
        for label in labels:
            sensor_name = str(label[0]).strip("[]'")
            table_name = f"{sensor_name}_{self.log_id}"
            
            # Usar o LogSeparator apenas para extrair os dados (sem salvar em CSV)
            separator = LogSeparator(label, self.log_filepath, data_subdir=None) # data_subdir não é usado
            data = separator.extract_data()
            
            if data:
                df = pd.DataFrame(data, columns=label[1])
                # Converte tipos de dados para SQL (importante!)
                df = df.apply(pd.to_numeric, errors='ignore')
                
                # Salva o DataFrame como uma nova tabela no SQL
                df.to_sql(table_name, self.conn, index=False, if_exists='replace')
                print(f"  -> Dados de '{sensor_name}' salvos na tabela '{table_name}'")

    def save_processed_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """
        Salva um DataFrame processado (ex: dados_variados) como uma tabela SQL.
        """
        # O 'filename' (ex: 'dados_variados.csv') vira o nome da tabela
        table_name_base = filename.split('.')[0]
        final_table_name = f"{table_name_base}_{self.log_id}"
        
        print(f"Salvando dados processados no SQL (Tabela: {final_table_name})...")
        df.to_sql(final_table_name, self.conn, index=False, if_exists='replace')
        print("Dados processados salvos.")

    def get_processed_dataframe(self, filename: str) -> pd.DataFrame:
        """
        Lê um DataFrame processado (ex: dados_variados) do SQL.
        """
        table_name_base = filename.split('.')[0]
        final_table_name = f"{table_name_base}_{self.log_id}"

        if not self._table_exists(final_table_name):
            msg = f"Tabela processada '{final_table_name}' não encontrada no SQL. Execute '/data' ou '/features' primeiro."
            print(msg)
            raise FileNotFoundError(msg)
        
        print(f"Carregando dados processados do SQL (Tabela: {final_table_name})...")
        return pd.read_sql(f"SELECT * FROM {final_table_name}", self.conn)

    def load_or_create_labels(self, extractor_func: Callable[[], Tuple[List[str], List[Any]]]) -> Tuple[List[str], List[Any]]:
        """
        Para este exemplo, o cache de labels (.npz) ainda é gerenciado
        pelo FileRepository, pois é mais simples.
        """
        print("Usando cache .npz para labels (via FileRepository)...")
        temp_file_repo = FileRepository(self.log_filepath)
        return temp_file_repo.load_or_create_labels(extractor_func)