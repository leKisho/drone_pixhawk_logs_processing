# domain/interfaces.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Tuple, Dict, Any, Callable

class ILogRepository(ABC):
    """
    Define o contrato para qualquer classe que gerencia o
    armazenamento e a recuperação de dados de voo.
    """

    @abstractmethod
    def load_or_create_labels(self, extractor_func: Callable[[], Tuple[List[str], List[Any]]]) -> Tuple[List[str], List[Any]]:
        """
        Tenta carregar os IDs e Labels do cache (.npz).
        Se não conseguir, executa a 'extractor_func' para parsear o log
        e salva o resultado no cache.
        """
        pass

    @abstractmethod
    def get_raw_sensor_data(self, required_sensors: Tuple[str, ...]) -> Dict[str, pd.DataFrame]:
        """
        Lê os arquivos CSV brutos (GPS, RFND, BARO, etc.) do diretório de dados
        e os retorna como um dicionário de DataFrames.
        """
        pass

    @abstractmethod
    def save_processed_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """
        Salva um DataFrame processado (ex: dados_variados.csv) no 
        diretório de dados.
        """
        pass

    @abstractmethod
    def get_processed_dataframe(self, filename: str) -> pd.DataFrame:
        """
        Lê um DataFrame já processado (ex: dados_variados.csv) do 
        diretório de dados. Usado pelo serviço de plotagem.
        """
        pass