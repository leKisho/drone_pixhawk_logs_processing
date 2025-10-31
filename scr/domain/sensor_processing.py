# domain/sensor_processing.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
from typing import Tuple, List, Any, Dict

# NOTA: O matplotlib é importado aqui condicionalmente para a função 'normalize'.
# Em uma refatoração "pura", essa visualização seria movida para a camada
# de Aplicação, mas mantemos aqui para preservar sua funcionalidade interativa.
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def align_sensor_data(
    gps_df: pd.DataFrame, 
    rfnd_df: pd.DataFrame, 
    baro_df: pd.DataFrame, 
    threshold: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: # <--- ATUALIZADO: Retorna 4 itens
    """
    (Lógica Pura movida de LogProcessor.z_filter)
    Alinha os dados do RFND e BARO com os timestamps do GPS usando uma janela
    temporal (threshold) e calcula a mediana.
    
    Retorna: z_agrupado, erros_tempo, amplitude_z, alt_agrupada
    """
    
    # 1. Prepara dados do RFND (marcando dados inválidos como NaN)
    rfnd_clean = rfnd_df.copy()
    if 'Stat1' in rfnd_clean.columns:
        mask = rfnd_clean['Stat1'] != 4
        rfnd_clean.loc[mask, 'Dist1'] = np.nan

    # 2. Prepara arrays para o loop (mais rápido que iterar DataFrames)
    tempos_gps = gps_df['TimeUS'].values
    tempos_z = rfnd_clean['TimeUS'].values
    tempos_baro = baro_df['TimeUS'].values

    valores_z = rfnd_clean['Dist1'].values
    valores_alt = baro_df['Alt'].values

    # 3. Listas para armazenar os resultados do alinhamento
    z_agrupado = []
    alt_agrupada = []
    erros_tempo = []
    # z_ls = []  <--- REMOVIDO
    amplitude_z = []

    print(f"Alinhando sensores com threshold de +/- {threshold:.0f} us...")
    
    # 4. Loop de alinhamento principal
    for t in tempos_gps:
        z_local = []
        alt_local = []
        menor = 10000000

        for tz, z in zip(tempos_z, valores_z):
            if abs(tz - t) <= threshold:
                z_local.append(z)
                if abs(tz - t) < menor:
                    menor = tz
        
        for ta, a in zip(tempos_baro, valores_alt):
            if abs(ta - t) <= threshold:
                alt_local.append(a)

        erros_tempo.append(abs(menor - t))

        # 5. Processa os dados da janela
        
        # Filtra NaNs (Stat1!=4) e Zeros (falha de leitura)
        z_filtrado = [v for v in z_local if pd.notna(v) and v > 0]
        # z_ls.append(z_filtrado)  <--- REMOVIDO      

        if z_filtrado:
            z_agrupado.append(np.median(z_filtrado))
            amplitude_z.append((abs(min(z_filtrado) - max(z_filtrado))) / 2)
        else:
            amplitude_z.append(0)
            z_agrupado.append(np.nan)

        if alt_local:
            alt_agrupada.append(np.median(alt_local))
        else:
            alt_agrupada.append(np.nan)
    
    print('Alinhamento concluído.')

    # Converte TODAS as saídas para pd.Series alinhadas ao índice
    index = gps_df.index
    return (
        pd.Series(z_agrupado, index=index, name="z_agrupado"),
        pd.Series(erros_tempo, index=index, name="erros_tempo"),
        # pd.Series(z_ls_serializada, ...)  <--- REMOVIDO
        pd.Series(amplitude_z, index=index, name="amplitude_z"),
        pd.Series(alt_agrupada, index=index, name="alt_agrupada")
    )

def normalize_signal_with_polynomial_fit(
    tempo: pd.Series, 
    alt: pd.Series,
    user_interaction: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    (Lógica Pura movida de LogProcessor.normalize)
    Encontra a linha de tendência (chão) de um sinal ruidoso usando
    detecção de picos e regressão polinomial.
    """
    
    # --- 1. ENCONTRAR MÍNIMOS INICIAIS ---
    alt_valid = alt.dropna()
    sinal_invertido = -alt_valid
    if sinal_invertido.empty:
        print("Aviso: Sinal 'alt' está vazio ou todo NaN. Retornando sinal original.")
        zeros = pd.Series(np.zeros_like(alt, dtype=float), index=tempo.index)
        return alt, zeros

    indices_minimos, _ = find_peaks(sinal_invertido, distance=80)
    
    # Mapeia os índices do sinal filtrado de volta para o original
    tempo_minimos = tempo.loc[sinal_invertido.index[indices_minimos]]
    altura_minimos = alt.loc[sinal_invertido.index[indices_minimos]]

    # --- 2. REMOÇÃO AUTOMÁTICA (MEDIANA MÓVEL) ---
    tamanho_janela = 11
    limiar_tolerancia = 5.0

    mediana_movel = altura_minimos.rolling(window=tamanho_janela, center=True, min_periods=1).median()
    diferenca_mediana = np.abs(altura_minimos - mediana_movel)
    mascara_bons_auto = diferenca_mediana < limiar_tolerancia

    tempo_minimos_auto = tempo_minimos[mascara_bons_auto]
    altura_minimos_auto = altura_minimos[mascara_bons_auto]
    tempo_outliers_auto = tempo_minimos[~mascara_bons_auto]
    altura_outliers_auto = altura_minimos[~mascara_bons_auto]
    
    print(f"Filtro automático removeu {len(tempo_outliers_auto)} pontos.")
    
    tempo_minimos_final = tempo_minimos_auto.to_numpy()
    altura_minimos_final = altura_minimos_auto.to_numpy()

    # --- 3. INTERAÇÃO HUMANA (Controlada pela Camada de Aplicação) ---
    if user_interaction:
        if plt is None:
            print("Aviso: Matplotlib não encontrado. Pulando plot interativo.")
        else:
            tempo_minimos_auto_np = tempo_minimos_auto.to_numpy()
            altura_minimos_auto_np = altura_minimos_auto.to_numpy()

            plt.figure(figsize=(18, 8))
            plt.plot(tempo, alt, label='Sinal Original', alpha=0.3)
            plt.scatter(tempo_minimos_auto, altura_minimos_auto, color='red', label='Mínimos Pós-Filtro Automático')
            
            for i, (tempo_val, altura_val) in enumerate(zip(tempo_minimos_auto, altura_minimos_auto)):
                plt.text(tempo_val, altura_val - 2, str(i), fontsize=9, ha='center', color='blue')

            plt.title('INSPEÇÃO: Digite os índices (em azul) que deseja remover')
            plt.xlabel('Tempo (TimeUS)'); plt.ylabel('Altura/Distância'); plt.legend(); plt.grid(True, alpha=0.3)
            plt.show(block=True)

            indices_str = input("Quais pontos deseja remover? Separe por vírgula (ex: 3,15). Deixe vazio para nenhum: ").split(',')
            
            indices_para_remover_manualmente = []
            max_indice_valido = len(altura_minimos_auto_np) - 1

            if indices_str and indices_str[0] != '':
                for val_str in indices_str:
                    try:
                        val = int(val_str.strip())
                        if 0 <= val <= max_indice_valido:
                            indices_para_remover_manualmente.append(val)
                        else:
                            print(f"AVISO: O índice {val} é inválido. Índices válidos são de 0 a {max_indice_valido}.")
                    except ValueError:
                        print(f"AVISO: A entrada '{val_str}' não é um número válido.")
            
            # Guarda os outliers manuais para o plot
            tempo_outliers_manual = tempo_minimos_auto_np[indices_para_remover_manualmente]
            altura_outliers_manual = altura_minimos_auto_np[indices_para_remover_manualmente]

            # Aplica o filtro final
            tempo_minimos_final = np.delete(tempo_minimos_auto_np, indices_para_remover_manualmente)
            altura_minimos_final = np.delete(altura_minimos_auto_np, indices_para_remover_manualmente)
            print(f"Filtro manual removeu mais {len(indices_para_remover_manualmente)} pontos.")

    print(f"Total de {len(altura_minimos_final)} mínimos restantes para a regressão.")

    # --- 4. REGRESSÃO POLINOMIAL ---
    grau_polinomio = 8
    # Garante que a linha de tendência tenha o mesmo índice que o tempo original
    linha_tendencia_polinomial = pd.Series(np.nan, index=tempo.index, dtype=float)

    if len(altura_minimos_final) > grau_polinomio:
        coeficientes = np.polyfit(tempo_minimos_final, altura_minimos_final, grau_polinomio)
        polinomio = np.poly1d(coeficientes)
        
        # Avalia o polinômio apenas em pontos válidos para evitar erros
        tempo_valido_index = tempo.dropna().index
        linha_tendencia_polinomial.loc[tempo_valido_index] = polinomio(tempo.loc[tempo_valido_index])
    else:
        print("Não há pontos suficientes para a regressão após a filtragem.")
        linha_tendencia_polinomial = pd.Series(np.zeros_like(tempo, dtype=float), index=tempo.index)

    # --- 5. NORMALIZAÇÃO ---
    RFND_Normalizado_Poli = alt - linha_tendencia_polinomial

    # --- 6. VISUALIZAÇÃO (Se interativo) ---
    if user_interaction and plt is not None:
        plt.figure(figsize=(15, 6))
        plt.plot(tempo, alt, label='Sinal Original (RFND)', alpha=0.5)
        plt.scatter(tempo_outliers_auto, altura_outliers_auto, color='gray', label='Outliers (Automático)', zorder=4)
        if 'tempo_outliers_manual' in locals():
             plt.scatter(tempo_outliers_manual, altura_outliers_manual, marker='x', color='orange', s=100, label='Outliers (Manual)', zorder=5)
        plt.scatter(tempo_minimos_final, altura_minimos_final, color='red', label='Mínimos Finais', zorder=6)
        plt.plot(tempo, linha_tendencia_polinomial, color='black', linestyle='--', linewidth=2, label=f'Tendência Polinomial (Grau {grau_polinomio})')
        plt.title('Detecção de Mínimos com Filtragem'); plt.xlabel('Tempo (TimeUS)'); plt.ylabel('Altura/Distância'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.show(block=True)

        plt.figure(figsize=(15, 6))
        plt.plot(tempo, RFND_Normalizado_Poli, label='Sinal Normalizado', color='purple')
        plt.axhline(0, color='black', linestyle='--')
        plt.title('Sinal RFND Normalizado'); plt.xlabel('Tempo (TimeUS)'); plt.ylabel('Variação de Altura/Distância'); plt.grid(True, alpha=0.3)
        plt.show(block=True)

    return RFND_Normalizado_Poli, linha_tendencia_polinomial

def filter_outliers_zscore(data: pd.Series, threshold: float = 2.0) -> pd.Series:
    """
    (Lógica movida de LogProcessor.data_preproxs)
    Aplica um filtro Z-score a uma Série Pandas, substituindo outliers por NaN.
    """
    z_scores = stats.zscore(np.nan_to_num(data))
    mascara_outliers = np.abs(z_scores) > threshold
    
    data_filtrada = data.copy()
    data_filtrada[mascara_outliers] = np.nan
    
    outliers_encontrados = np.sum(mascara_outliers)
    print(f"Método Z-score encontrou e removeu {outliers_encontrados} outliers.")
    
    return data_filtrada