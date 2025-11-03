# plot_comparacao_terreno_3d.py
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks

# Importa o Repositório da sua arquitetura
from scr.infrastructure.sql_repository import SQLRepository
from scr.infrastructure.file_repository import FileRepository

# -----------------------------------------------------------------
# --- CONFIGURAÇÃO ---
# (Copie as configurações relevantes do seu main.py)
# -----------------------------------------------------------------
LOG_FILE_PATH = "./assets/logs/2025-08-22 11-05-54.log"
USE_SQL_DATABASE = True
DATABASE_NAME = "logs_db.db"
GRAU_POLINOMIO = 8 # Grau para a regressão dos picos do LIDAR
# -----------------------------------------------------------------

def estimate_ground_trend_from_lidar(
    tempo: pd.Series, 
    lidar_distance: pd.Series,
    grau_polinomio: int = 8
) -> pd.Series:
    """
    (Função local para este script de validação)
    Estima a linha de tendência do "chão" a partir dos dados do LIDAR,
    encontrando os picos (MÁXIMOS) de distância.
    
    Retorna: A linha de tendência da distância do chão.
    """
    print("Estimando tendência do chão (solo) a partir dos picos do LIDAR...")
    
    sinal_valido = lidar_distance.dropna() 
    
    if sinal_valido.empty:
        print("Aviso: Sinal LIDAR está vazio. Retornando tendência vazia.")
        return pd.Series(np.nan, index=tempo.index, dtype=float)

    indices_maximos, _ = find_peaks(sinal_valido, distance=80)
    
    tempo_maximos = tempo.loc[sinal_valido.index[indices_maximos]]
    distancia_maximos = lidar_distance.loc[sinal_valido.index[indices_maximos]]

    tamanho_janela = 11
    limiar_tolerancia = 5.0 

    mediana_movel = distancia_maximos.rolling(window=tamanho_janela, center=True, min_periods=1).median()
    diferenca_mediana = np.abs(distancia_maximos - mediana_movel)
    mascara_bons_auto = diferenca_mediana < limiar_tolerancia

    tempo_maximos_filtrados = tempo_maximos[mascara_bons_auto].to_numpy()
    distancia_maximos_filtrados = distancia_maximos[mascara_bons_auto].to_numpy()
    
    print(f"Total de {len(distancia_maximos_filtrados)} picos (chão) restantes para a regressão.")

    linha_tendencia_chao = pd.Series(np.nan, index=tempo.index, dtype=float)

    if len(distancia_maximos_filtrados) > grau_polinomio:
        coeficientes = np.polyfit(tempo_maximos_filtrados, distancia_maximos_filtrados, grau_polinomio)
        polinomio = np.poly1d(coeficientes)
        
        tempo_valido_index = tempo.dropna().index
        linha_tendencia_chao.loc[tempo_valido_index] = polinomio(tempo.loc[tempo_valido_index])
    else:
        print("Aviso: Não há pontos de pico suficientes para a regressão do chão do LIDAR.")
        linha_tendencia_chao = pd.Series(np.nan, index=tempo.index, dtype=float)

    linha_tendencia_chao = linha_tendencia_chao.interpolate(method='linear').ffill().bfill()

    print("Tendência do chão do LIDAR estimada.")
    return linha_tendencia_chao

def main():
    """
    Script standalone para plotar uma comparação 3D de todas
    as fontes de altitude: GPS, TERR e LIDAR (Picos).
    """
    print("--- Plotter 3D de Comparação de Terreno ---")
    
    # 1. Validar arquivo de log
    if not os.path.exists(LOG_FILE_PATH):
        print(f"Erro: Arquivo de log não encontrado em {LOG_FILE_PATH}")
        sys.exit(1)
        
    print(f"Usando log: {LOG_FILE_PATH}")

    # 2. Instanciar o Repositório
    if USE_SQL_DATABASE:
        print(f"Usando banco de dados: {DATABASE_NAME}")
        repo = SQLRepository(LOG_FILE_PATH, db_name=DATABASE_NAME)
    else:
        print("Usando modo de arquivos CSV.")
        repo = FileRepository(LOG_FILE_PATH)

    # 3. Carregar os dados PROCESSADOS
    data_key = "dados_variados.csv"
    try:
        print(f"Carregando dados processados (chave: {data_key})...")
        df = repo.get_processed_dataframe(data_key)
    except FileNotFoundError as e:
        print(f"\nErro: {e}")
        print("Certifique-se de que 'main.py' já foi executado pelo menos uma vez para este log,")
        print("pois ele é necessário para gerar a tabela 'dados_variados'.")
        sys.exit(1)
    except Exception as e:
        print(f"Um erro inesperado ocorreu ao conectar ao banco: {e}")
        sys.exit(1)
        
    # 4. Validar e Limpar os Dados
    # Colunas necessárias:
    # 'x', 'y' (UTM)
    # 'Alt GPS' (Caminho do drone)
    # 'Alt_Solo_TERR' (Chão do TERR)
    # 'TimeUS', 'Dist' (Necessários para calcular o chão do LIDAR)
    colunas_necessarias = ['x', 'y', 'Alt GPS', 'Alt_Solo_TERR', 'TimeUS', 'Dist']
    
    if not all(col in df.columns for col in colunas_necessarias):
        print(f"Erro: O dataframe 'dados_variados' não contém as colunas necessárias.")
        print(f"Necessárias: {colunas_necessarias}")
        print(f"Encontradas: {df.columns.tolist()}")
        sys.exit(1)
        
    df_cleaned = df.copy()
    
    for col in colunas_necessarias:
         df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
         
    df_cleaned = df_cleaned.dropna(subset=colunas_necessarias)

    if df_cleaned.empty:
        print("Erro: Não há dados válidos para plotar.")
        sys.exit(1)
        
    # 5. Executar a nova lógica de estimativa (localmente)
    tendencia_chao_lidar = estimate_ground_trend_from_lidar(
        df_cleaned['TimeUS'], 
        df_cleaned['Dist'], 
        grau_polinomio=GRAU_POLINOMIO
    )
    
    # Calcula a altitude do chão estimada pelo LIDAR
    df_cleaned['Alt_Solo_LIDAR'] = df_cleaned['Alt GPS'] - tendencia_chao_lidar
    
    print(f"Plotando {len(df_cleaned)} pontos de dados 3D...")

    # 6. Plotar
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 1: O caminho do drone (GPS)
    ax.plot(df_cleaned['x'], df_cleaned['y'], df_cleaned['Alt GPS'], 
            label='Caminho do Drone (Alt GPS)', color='blue', alpha=0.7, linewidth=2)
    
    # Plot 2: O terreno (TERR)
    ax.scatter(df_cleaned['x'], df_cleaned['y'], df_cleaned['Alt_Solo_TERR'], 
               label='Terreno (Alt_Solo_TERR)', color='green', s=3, alpha=0.5)
    
    # Plot 3: O terreno (LIDAR Picos)
    ax.scatter(df_cleaned['x'], df_cleaned['y'], df_cleaned['Alt_Solo_LIDAR'], 
               label='Terreno (LIDAR Picos)', color='red', s=3, alpha=0.5)
    
    ax.set_title(f'Comparação 3D das Fontes de Altitude\nLog: {repo.log_id}')
    ax.set_xlabel('X (UTM)')
    ax.set_ylabel('Y (UTM)')
    ax.set_zlabel('Altitude (m)')
    ax.legend()
    
    print("Exibindo gráfico... Feche a janela para sair.")
    plt.show()

if __name__ == "__main__":
    main()