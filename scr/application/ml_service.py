# scr/application/ml_service.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os
import sqlite3 # Necessário para escanear o DB

# Importar a interface do repositório
from scr.domain.interfaces import ILogRepository

# Importa o SQLRepository SÓ PARA O TYPE HINTING, para acessar o 'db_path'
# Isso é um "code smell" leve, mas necessário para a lógica de "treinamento global"
from scr.infrastructure.sql_repository import SQLRepository

class MLService:
    
    def __init__(self, repo: ILogRepository, model_path: str, gabarito_key: str):
        self.repo = repo
        self.model_path = model_path
        self.gabarito_prefix = gabarito_key.split('.')[0] # ex: 'gabarito_programatico'
        self.features = ['z_median', 'z_std_dev', 'z_amplitude', 'z_percent_falhas', 'alt_baro']

    def _get_all_gabarito_tables(self) -> list:
        """
        Conecta-se ao banco de dados do 'repo' e encontra TODAS as tabelas
        que começam com o prefixo do gabarito (ex: 'gabarito_programatico_').
        """
        
        # Esta é a única forma de fazer isso funcionar com o FileRepository
        if not isinstance(self.repo, SQLRepository):
            print("Aviso: Treinamento global com FileRepository usa apenas o gabarito do log atual.")
            # Se for FileRepo, só existe um gabarito
            key = f"{self.gabarito_prefix}_{self.repo.log_id}"
            return [key]

        # Se for SQLRepository, escaneia o DB
        db_path = self.repo.db_path
        if not os.path.exists(db_path):
            return []
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Encontra todas as tabelas que começam com o prefixo
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '{self.gabarito_prefix}_%';")
            
            table_names = [item[0] for item in cursor.fetchall()]
            conn.close()
            return table_names
        except Exception as e:
            print(f"Erro ao escanear tabelas do DB: {e}")
            return []

    def train_global_model(self):
        """
        Carrega TODOS os gabaritos do banco de dados ÚNICO, 
        os combina e treina um modelo global.
        """
        print(f"Iniciando treinamento do modelo GLOBAL...")
        
        # 1. Encontrar todas as tabelas de gabarito no DB
        gabarito_tables = self._get_all_gabarito_tables()
        
        if not gabarito_tables:
            print(f"Erro: Nenhum gabarito encontrado em '{self.repo.db_path}' com prefixo '{self.gabarito_prefix}_'.")
            print("Execute 'python main.py --label' em um ou mais logs primeiro.")
            return

        print(f"Encontrados {len(gabarito_tables)} gabaritos para treinamento global:")
        
        # 2. Carregar e concatenar todos os gabaritos
        all_data = []
        for table_name in gabarito_tables:
            print(f"  - Carregando {table_name}...")
            try:
                # O repo (SQL) está conectado ao DB certo
                # Usamos a conexão interna do repo para ler a tabela
                df = pd.read_sql(f"SELECT * FROM {table_name}", self.repo.conn)
                all_data.append(df)
            except Exception as e:
                print(f"    Aviso: Falha ao carregar {table_name}. Erro: {e}")
        
        if not all_data:
            print("Erro: Falha ao carregar todos os gabaritos.")
            return

        df_combined = pd.concat(all_data, ignore_index=True)

        # 3. Preparar dados combinados
        df_labeled = df_combined.dropna(subset=['Classe'])
        df_labeled = df_labeled.dropna(subset=self.features)
        
        if len(df_labeled) < 50:
            print("Erro: Dados rotulados insuficientes. Rotule mais linhas.")
            return

        print(f"\nTotal de {len(df_labeled)} amostras combinadas para treinamento.")

        X = df_labeled[self.features]
        y = df_labeled['Classe'].astype(int)

        # 4. Dividir em treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # 5. Treinar o modelo
        print("Treinando RandomForestClassifier (Modelo Global)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # 6. Avaliar o modelo
        print("Avaliação do modelo GLOBAL nos dados de teste:")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        print("Importância das features (Global):")
        for feature, importance in zip(self.features, model.feature_importances_):
            print(f"  {feature}: {importance:.3f}")

        # 7. Salvar o modelo
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        print(f"Modelo GLOBAL salvo com sucesso em: {self.model_path}")

    # (O 'classify_data' não precisa mudar)
    def classify_data(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Carrega o modelo GLOBAL treinado e o usa para prever as classes
        de um novo DataFrame de features.
        """
        if not os.path.exists(self.model_path):
            print(f"Erro: Modelo GLOBAL não encontrado em {self.model_path}")
            print("Execute o treinamento ('python main.py') primeiro.")
            features_df['Classe_Predita'] = -1 
            return features_df

        print(f"Carregando modelo GLOBAL de: {self.model_path}")
        model = joblib.load(self.model_path)
        
        X_predict = features_df[self.features].fillna(0)
        
        print("Classificando dados do voo com modelo GLOBAL...")
        predictions = model.predict(X_predict)
        
        features_df['Classe_Predita'] = predictions
        print("Classificação concluída.")
        
        return features_df