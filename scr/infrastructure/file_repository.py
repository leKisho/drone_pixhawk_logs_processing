# infrastructure/file_repository.py
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Callable
from .interfaces import ILogRepository

class FileRepository(ILogRepository):
    """
    Implementação concreta do Repositório.
    Lida com todo o acesso a arquivos: .npz (cache) e .csv (dados).
    """
    def __init__(self, log_filepath: str):
        self.log_filepath = log_filepath
        self.log_dir = os.path.dirname(log_filepath)
        
        # Define o nome do arquivo de log sem extensão (ex: '2025-09-03 11-30-05')
        self.log_filename_no_ext = os.path.basename(log_filepath).split('.')[0]
        
        # O subdiretório onde todos os dados CSV são salvos
        # ex: './2025-09-03 11-30-05/'
        self.data_subdir = os.path.join(self.log_dir, self.log_filename_no_ext)
        
        # O arquivo de cache .npz para os labels
        # ex: './2025-09-03 11-30-05.log.npz'
        self.cache_file = os.path.join(self.log_dir, f"{os.path.basename(log_filepath)}.npz")

    def load_or_create_labels(self, extractor_func: Callable[[], Tuple[List[str], List[Any]]]) -> Tuple[List[str], List[Any]]:
        """
        (Lógica movida de LogProcessor.load_or_label)
        Tenta carregar do cache .npz. Se falhar, executa a função
        de extração e salva o resultado no cache.
        """
        if os.path.exists(self.cache_file):
            print(f"Carregando labels do cache: {self.cache_file}")
            data = np.load(self.cache_file, allow_pickle=True)
            ids = list(data['id'])
            labels = list(data['label'])
            return ids, labels
        else:
            print("Cache não encontrado. Parseando arquivo .log...")
            # A 'extractor_func' é o método LogLabeler.extract_labels,
            # passado pela camada de aplicação.
            ids, labels = extractor_func()
            np.savez(self.cache_file, id=np.array(ids, dtype=object), label=np.array(labels, dtype=object))
            print(f"Labels salvos no cache: {self.cache_file}")
            return ids, labels

    def get_raw_sensor_data(self, required_sensors: Tuple[str, ...]) -> Dict[str, pd.DataFrame]:
        """
        (Lógica movida de LogProcessor.data_extract)
        Lê os CSVs brutos (GPS, RFND, BARO) do subdiretório de dados.
        """
        sensor_data: Dict[str, pd.DataFrame] = {}
        
        if not os.path.exists(self.data_subdir):
            print(f"Erro: Diretório de dados não encontrado: {self.data_subdir}")
            print("Execute o comando de separação de logs primeiro.")
            return sensor_data

        print(f"Lendo dados brutos de: {self.data_subdir}")
        for filename in os.listdir(self.data_subdir):
            # O ID do sensor está no nome do arquivo, ex: (nome).GPS.csv
            doc_id = filename.split('.')[-2]
            
            if doc_id in required_sensors:
                try:
                    filepath = os.path.join(self.data_subdir, filename)
                    sensor_data[doc_id] = pd.read_csv(filepath)
                except Exception as e:
                    print(f"Erro ao ler o arquivo {filename}: {e}")
                    
        return sensor_data

    def save_processed_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """
        (Lógica movida de LogProcessor.data_preproxs e outros)
        Salva um DataFrame processado no subdiretório de dados.
        """
        os.makedirs(self.data_subdir, exist_ok=True)
        output_path = os.path.join(self.data_subdir, filename)
        
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Arquivo processado salvo: {output_path}")
        except Exception as e:
            print(f"Erro ao salvar o arquivo {output_path}: {e}")

    def get_processed_dataframe(self, filename: str) -> pd.DataFrame:
        """
        (Lógica necessária para o PlottingService)
        Lê um DataFrame já processado do subdiretório de dados.
        """
        filepath = os.path.join(self.data_subdir, filename)
        if not os.path.exists(filepath):
            print(f"Erro: Arquivo processado não encontrado: {filepath}")
            print("Execute o comando de processamento ('/data' ou '/features') primeiro.")
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
            
        return pd.read_csv(filepath)