# infrastructure/log_parser.py
import os
import pandas as pd
from typing import List, Tuple, Any

class LogLabeler:
    """
    (Movido de log_reader 2.1.2.py)
    Lê um arquivo de log e extrai os cabeçalhos de formato (FMT).
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.ids: List[str] = []
        self.labels: List[Any] = []

    def extract_labels(self) -> Tuple[List[str], List[Any]]:
        with open(self.filepath, 'r') as log:
            for line in log:
                parts = line.split(',', 5)
                if parts[0] == "FMT":
                    ident = [parts[3][1:]]
                    fields = [ident, parts[5][1:].strip().split(',')]
                    self.ids.append(ident)
                    self.labels.append(fields)
        return self.ids, self.labels


class LogSeparator:
    """
    (Movido de log_reader 2.1.2.py)
    Lê um arquivo de log e salva os dados de um 'label' (ID)
    específico em um arquivo .csv separado.
    """
    def __init__(self, label: Any, filepath: str, data_subdir: str):
        self.label = label
        self.filepath = filepath
        self.data_subdir = data_subdir # Onde os CSVs devem ser salvos

    def extract_data(self) -> List[List[str]]:
        data = []
        with open(self.filepath, 'r') as log:
            for line in log:
                parts = line.split(',', len(self.label[1]))
                if parts[0] == str(self.label[0]).strip("[']"):
                    parts[-1] = parts[-1].replace('\n', '')
                    data.append(parts[1:])
        return data

    def save_to_csv(self) -> None:
        data = self.extract_data()
        
        # Garante que o subdiretório de dados (ex: '2025-09-03 11-30-05/') existe
        os.makedirs(self.data_subdir, exist_ok=True)
        
        label_c = str(self.label[0]).strip("[]\'")
        log_filename = os.path.basename(self.filepath).split('.')[0]
        filename = f'({log_filename}).{label_c}.csv'
        output_path = os.path.join(self.data_subdir, filename)
        
        df = pd.DataFrame(data, columns=self.label[1])
        if df.empty:
            print(f"Aviso: O ID {self.label[0]} gerou um dataframe vazio. Não será gerado um arquivo.\n")
        else:
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"Arquivo CSV salvo: {output_path}")