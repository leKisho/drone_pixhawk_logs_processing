# main.py
import os
import sys

from scr.application.services import ApplicationService
from scr.application.plotting_service import PlottingService
from scr.infrastructure.file_repository import FileRepository
from scr.infrastructure.sql_repository import SQLRepository

# -----------------------------------------------------------------
# --- CONFIGURAÇÃO GLOBAL ---
# -----------------------------------------------------------------
LOG_FILE_PATH = "./assets/logs/2025-09-03 11-30-05.log"
mins = 0
maxs = -1

# Mude para True para usar o Banco de Dados, False para usar CSVs
USE_SQL_DATABASE = True
DATABASE_NAME = "logs_db.db" # Nome do arquivo .db

# Mude para False para PULAR a etapa de normalização
APLICAR_CORRECAO_POLINOMIAL = True 

# Mude para False para rodar a correção sem os plots interativos
# (Isso só funciona se APLICAR_CORRECAO_POLINOMIAL for True)
CORRECAO_POLINOMIAL_INTERATIVA = True
# -----------------------------------------------------------------


if __name__ == "__main__":
    """
    Ponto de entrada principal da aplicação (modo script).
    """
    
    # 1. Validação de Arquivo
    if not os.path.exists(LOG_FILE_PATH):
        print(f"Erro Crítico: O arquivo de log principal não foi encontrado.")
        print(f"Localização esperada: {os.path.abspath(LOG_FILE_PATH)}")
        sys.exit(1)

    print(f"Arquivo de log carregado: {LOG_FILE_PATH}")

    # 2. Configuração (Injeção de Dependência)
    #    O main decide qual repositório será usado, com base na flag
    
    if USE_SQL_DATABASE:
        print(f"Modo de Repositório: SQL (Banco: {DATABASE_NAME})")
        repo = SQLRepository(LOG_FILE_PATH, db_name=DATABASE_NAME)
    else:
        print("Modo de Repositório: Arquivos (CSV)")
        repo = FileRepository(LOG_FILE_PATH)

    # --- NENHUM CÓDIGO ABAIXO DESTA LINHA PRECISA MUDAR ---
    # Os serviços 'plotter' e 'app_service' não sabem (e não se importam)
    # qual repositório estão usando. Eles apenas usam o "contrato".
    
    plotter = PlottingService(repo)
    app_service = ApplicationService(repo, LOG_FILE_PATH)

    print("\n--- Processador de Log de Voo ---")
    print("Executando tarefas em main.py...")

    # 3. Execução das Tarefas
    try:
        # --- Tarefa 1: Separar os logs (se necessário) ---
        # No modo SQL, esta etapa é "preguiçosa". Ela só será executada
        # pelo 'get_raw_sensor_data' se os dados ainda não estiverem no banco.
        # No modo CSV, ela precisa ser executada.
        if not USE_SQL_DATABASE:
            print("Modo CSV: Executando separação de logs...")
            ids_para_separar = "ahr2,baro,cam,gps,pos,rfnd,terr,guid,orgn,parm,bat,cmd,err,ev,mode,msg".split(',')
            app_service.process_logs_to_csv(id_list=ids_para_separar)
        else:
            print("Modo SQL: Separação de logs será automática na primeira chamada de /data (se necessário).")

        # --- Tarefa 2: Processar os dados principais ---
        # MODIFIQUE ESTA CHAMADA para passar as flags
        app_service.process_main_data(
            mins=mins, 
            maxs=maxs,
            aplicar_correcao=APLICAR_CORRECAO_POLINOMIAL,
            modo_interativo=CORRECAO_POLINOMIAL_INTERATIVA
        )
        
        # --- Tarefa 3: Plotar ---
        
        #plotter.run_plot("3d_profiles")
        #plotter.run_plot("terr_alt")

        # (Descomente para rodar outros comandos)
        #app_service.generate_ml_features()
        #plotter.run_plot("ml_dashboard")
        #plotter.run_plot("outlier_analysis")

        print("\n--- Execução do script concluída ---")

    except Exception as e:
        print(f"\n--- ERRO INESPERADO ---")
        print(f"Ocorreu um erro durante a execução do script:")
        print(f"Tipo: {type(e).__name__}")
        print(f"Mensagem: {e}")
        import traceback
        traceback.print_exc()