# main.py
import os
import sys

# --- Importa os Serviços da Camada de Aplicação ---
from scr.application.services import ApplicationService
from scr.application.plotting_service import PlottingService
from scr.application.ml_service import MLService 

# --- Importa AMBAS as Implementações de Repositório ---
from scr.infrastructure.file_repository import FileRepository
from scr.infrastructure.sql_repository import SQLRepository

from scr.domain.interfaces import ILogRepository

# -----------------------------------------------------------------
# --- CONFIGURAÇÃO GLOBAL ---
# (Suas configurações - ESTÃO PERFEITAS)
# -----------------------------------------------------------------
LOG_FILE_PATH = "./assets/logs/2025-09-03 11-30-05.log"
mins = 0
maxs = -1

# --- Configuração do Repositório (ÚNICO) ---
USE_SQL_DATABASE = True
DATABASE_NAME = "logs_db.db" 

# --- Configurações do Processamento de Dados (/data) ---
APLICAR_CORRECAO_POLINOMIAL = True 
CORRECAO_POLINOMIAL_INTERATIVA = True 
GRAU_POLINOMIO_INICIAL = 8

# --- Configurações de Machine Learning ---
GABARITO_KEY_NAME = "gabarito_programatico.csv" 
MODELO_PATH = "./assets/models/rf_classifier_2class.joblib"
TREINAR_MODELO = True 
CLASSIFICAR_DADOS = True

# --- CONFIGURAÇÃO DE LABELING ---
LABELING_ALTITUDE_SOURCE = "barometer"
# -----------------------------------------------------------------


def run_main_processing_pipeline(repo: ILogRepository, app_service: ApplicationService, plotter: PlottingService, ml_service: MLService):
    """
    Orquestra o pipeline de processamento de dados principal.
    (Executado por padrão ou com 'python main.py')
    """
    print("\n--- Processador de Log de Voo (Modo: Processamento) ---")
    try:
        # --- Tarefa 1: Separar os logs (se necessário) ---
        if not USE_SQL_DATABASE:
            print("Modo CSV: Executando separação de logs...")
            app_service.process_logs_to_csv()
        else:
            print("Modo SQL: Separação de logs será automática...")

        # --- Tarefa 2: Processar os dados principais (/data) ---
        print("\nProcessando dados principais (/data)...")
        app_service.process_main_data(
            mins=mins, 
            maxs=maxs,
            aplicar_correcao=APLICAR_CORRECAO_POLINOMIAL,
            modo_interativo=CORRECAO_POLINOMIAL_INTERATIVA,
            grau_polinomio=GRAU_POLINOMIO_INICIAL
        )
        
        # --- Tarefa 3: Plotar resultados do /data ---
        #plotter.run_plot("3d_profiles")
        #plotter.run_plot("terr_alt")
        plotter.run_plot("terr_profile") # Adicionando o último plot que pedimos

        # --- Tarefa 4: Gerar Features de ML (/features) ---
        print("\n--- Iniciando Etapa de ML ---")
        
        # (DESCOMENTADO: Esta linha é NECESSÁRIA para o ML funcionar)
        app_service.generate_ml_features() 
        plotter.run_plot("ml_dashboard")
        
        # --- Tarefa 5: Treinar e Classificar (ML) ---
        if TREINAR_MODELO:
            print(f"\nTreinando modelo GLOBAL com dados de: {DATABASE_NAME}")
            # O ml_service (usando o 'repo' único) vai
            # escanear o DB por todas as tabelas 'gabarito_...'
            ml_service.train_global_model() 
        
        if CLASSIFICAR_DADOS:
            try:
                print("\nCarregando 'features_ml' para classificação...")
                features_para_classificar = repo.get_processed_dataframe("features_ml.csv")
                df_classificado = ml_service.classify_data(features_para_classificar)
                
                repo.save_processed_dataframe(df_classificado, "classified_data.csv")
                print("Dados classificados salvos no repositório (tabela: classified_data_...).")
                
                plotter.run_plot("classification_map")
                
            except FileNotFoundError as e:
                print(f"Erro: Não foi possível carregar 'features_ml.csv' para classificação. {e}")
            
        print("\n--- Execução do script principal concluída ---")

    except Exception as e:
        print(f"\n--- ERRO INESPERADO (PIPELINE PRINCIPAL) ---")
        print(f"Ocorreu um erro durante a execução do script:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensagem: {e}")
        import traceback
        traceback.print_exc()


def run_interactive_labeling(repo: ILogRepository, app_service: ApplicationService, plotter: PlottingService):
    """
    Orquestra o pipeline de criação de gabarito interativo.
    (Executado com 'python main.py --label')
    """
    print("\n--- Processador de Log de Voo (Modo: Labeling Interativo) ---")
    print("Este assistente irá gerar o gabarito (labels) para o ML.")
    
    try:
        # 1. Chama o Caso de Uso, injetando o plotter
        df_gabarito = app_service.generate_programmatic_labels(
            plotter, 
            labeling_source=LABELING_ALTITUDE_SOURCE
        )
        
        if df_gabarito is not None:
            # 2. O 'repo' (único) salva o gabarito no DB
            repo.save_processed_dataframe(df_gabarito, GABARITO_KEY_NAME)
            
            print("\n--- Concluído! ---")
            print(f"Gabarito programático salvo no REPOSITÓRIO (chave: {GABARITO_KEY_NAME})")
            print(f"(Tabela: {GABARITO_KEY_NAME.split('.')[0]}_{repo.log_id})")
            print("\nDistribuição das classes criadas (0=Solo/Água, 1=Vegetação):")
            print(df_gabarito['Classe'].value_counts(dropna=False).to_string())
            print(f"\nPróximo passo: Ajuste 'TREINAR_MODELO = True' e rode 'python main.py' para treinar.")
            
    except Exception as e:
        print(f"\n--- ERRO INESPERADO (LABELING) ---")
        print(f"Ocorreu um erro: {e}")
        print("Certifique-se que 'features_ml.csv' existe no repositório (execute 'python main.py' uma vez primeiro).")
        import traceback
        traceback.print_exc()


# -----------------------------------------------------------------
# --- BLOCO PRINCIPAL (CORRIGIDO) ---
# -----------------------------------------------------------------
if __name__ == "__main__":
    """
    Ponto de entrada principal da aplicação (Composition Root).
    """
    
    # 1. Validação de Arquivo
    if not os.path.exists(LOG_FILE_PATH):
        print(f"Erro Crítico: O arquivo de log principal não foi encontrado.")
        print(f"Localização esperada: {os.path.abspath(LOG_FILE_PATH)}")
        sys.exit(1)

    print(f"Arquivo de log carregado: {LOG_FILE_PATH}")

    # 2. Configuração (Injeção de Dependência)
    # Instancia o ÚNICO repositório
    if USE_SQL_DATABASE:
        print(f"Repositório Único: SQL (Banco: {DATABASE_NAME})")
        repo = SQLRepository(LOG_FILE_PATH, db_name=DATABASE_NAME)
    else:
        print("Repositório Único: Arquivos (CSV)")
        repo = FileRepository(LOG_FILE_PATH)

    # Instancia todos os serviços, injetando o MESMO repo
    plotter = PlottingService(repo)
    app_service = ApplicationService(repo, LOG_FILE_PATH)
    ml_service = MLService(repo=repo, 
                           model_path=MODELO_PATH, 
                           gabarito_key=GABARITO_KEY_NAME)
    
    
    # 3. Decidir o modo de operação
    
 
    # Se o usuário quer fazer o processamento principal
    run_main_processing_pipeline(repo, app_service, plotter, ml_service)

    # Se o usuário quer fazer labeling interativo
    #run_interactive_labeling(repo, app_service, plotter)