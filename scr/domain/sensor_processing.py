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
    user_interaction: bool = True,
    initial_degree: int = 8
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
    
    # Estes são os pontos que usaremos para o fit, e que serão modificados interativamente
    tempo_minimos_para_fit = tempo_minimos_auto.to_numpy()
    altura_minimos_para_fit = altura_minimos_auto.to_numpy()

    # Guarda os outliers manuais para o plot final
    tempo_outliers_manual = np.array([])
    altura_outliers_manual = np.array([])
    
    grau_polinomio = initial_degree # Usa o grau vindo do main.py

    # --- 3. INTERAÇÃO HUMANA (Refatorado com Loops) ---
    if user_interaction:
        if plt is None:
            print("Aviso: Matplotlib não encontrado. Pulando plot interativo.")
        else:
            
            # --- INÍCIO DO LOOP 1: AJUSTE DO GRAU DO POLINÔMIO ---
            while True:
                print(f"--- Ajuste da Regressão Polinomial (Grau Atual: {grau_polinomio}) ---")

                # 4. REGRESSÃO (DENTRO DO LOOP)
                linha_tendencia_polinomial = pd.Series(np.nan, index=tempo.index, dtype=float)

                if len(altura_minimos_para_fit) > grau_polinomio:
                    coeficientes = np.polyfit(tempo_minimos_para_fit, altura_minimos_para_fit, grau_polinomio)
                    polinomio = np.poly1d(coeficientes)
                    tempo_valido_index = tempo.dropna().index
                    linha_tendencia_polinomial.loc[tempo_valido_index] = polinomio(tempo.loc[tempo_valido_index])
                else:
                    print(f"Aviso: Não há pontos suficientes ({len(altura_minimos_para_fit)}) para o grau {grau_polinomio}.")
                    linha_tendencia_polinomial = pd.Series(np.zeros_like(tempo, dtype=float), index=tempo.index)

                # 6. VISUALIZAÇÃO (DENTRO DO LOOP)
                plt.figure(figsize=(15, 6))
                plt.title(f'Visualização do Polinômio (Grau {grau_polinomio})')
                plt.plot(tempo, alt, label='Sinal Original (RFND)', alpha=0.5)
                plt.scatter(tempo_outliers_auto, altura_outliers_auto, color='gray', label='Outliers (Automático)', zorder=4)
                plt.scatter(tempo_minimos_para_fit, altura_minimos_para_fit, color='red', label='Mínimos (para Fit)', zorder=6)
                plt.plot(tempo, linha_tendencia_polinomial, color='black', linestyle='--', linewidth=2, label=f'Tendência Polinomial (Grau {grau_polinomio})')
                plt.legend(); plt.grid(True, alpha=0.3)
                plt.show(block=True)
                
                # PERGUNTA SOBRE O GRAU
                resposta_grau = input(f"O grau do polinômio ({grau_polinomio}) está adequado? (s/n): ").lower().strip()
                
                if resposta_grau == 's' or resposta_grau == 'sim':
                    break # Sai do loop do grau
                else:
                    try:
                        novo_grau = int(input("Digite o novo grau do polinômio (ex: 6): "))
                        grau_polinomio = novo_grau
                    except ValueError:
                        print("Entrada inválida. Por favor, digite um número inteiro.")
            # --- FIM DO LOOP 1 ---

            # --- INÍCIO DO LOOP 2: REMOÇÃO DE PONTOS ---
            while True:
                plt.figure(figsize=(18, 8))
                plt.plot(tempo, alt, label='Sinal Original', alpha=0.3)
                
                # Mostra outliers já removidos
                plt.scatter(tempo_outliers_auto, altura_outliers_auto, color='gray', label='Outliers (Automático)', zorder=4)
                if len(tempo_outliers_manual) > 0:
                    plt.scatter(tempo_outliers_manual, altura_outliers_manual, marker='x', color='orange', s=100, label='Outliers (Manual)', zorder=5)
                
                # Mostra pontos restantes com seus índices
                for i, (tempo_val, altura_val) in enumerate(zip(tempo_minimos_para_fit, altura_minimos_para_fit)):
                    plt.scatter(tempo_val, altura_val, color='red')
                    plt.text(tempo_val, altura_val - 2, str(i), fontsize=9, ha='center', color='blue')
                
                # Recalcula e plota a curva COM os pontos atuais
                if len(altura_minimos_para_fit) > grau_polinomio:
                    coeficientes = np.polyfit(tempo_minimos_para_fit, altura_minimos_para_fit, grau_polinomio)
                    polinomio = np.poly1d(coeficientes)
                    tempo_valido_index = tempo.dropna().index
                    linha_tendencia_polinomial.loc[tempo_valido_index] = polinomio(tempo.loc[tempo_valido_index])
                    plt.plot(tempo, linha_tendencia_polinomial, color='black', linestyle='--', linewidth=2, label=f'Tendência (Grau {grau_polinomio})')
                
                plt.scatter([], [], color='red', label='Mínimos Finais (para Fit)') # Apenas para a legenda
                plt.title('INSPEÇÃO: Digite os índices (em azul) que deseja remover')
                plt.xlabel('Tempo (TimeUS)'); plt.ylabel('Altura/Distância'); plt.legend(); plt.grid(True, alpha=0.3)
                plt.show(block=True)

                indices_str = input("Quais pontos deseja remover? Separe por vírgula (ex: 3,15). Deixe vazio para terminar: ").split(',')
                
                if not indices_str or indices_str[0] == '':
                    print("Nenhum ponto removido. Finalizando seleção.")
                    break # Sai do loop de remoção de pontos
                
                indices_para_remover = []
                max_indice_valido = len(altura_minimos_para_fit) - 1

                for val_str in indices_str:
                    try:
                        val = int(val_str.strip())
                        if 0 <= val <= max_indice_valido:
                            indices_para_remover.append(val)
                        else:
                            print(f"AVISO: O índice {val} é inválido. Índices válidos são de 0 a {max_indice_valido}.")
                    except ValueError:
                        print(f"AVISO: A entrada '{val_str}' não é um número válido.")
                
                if not indices_para_remover:
                    print("Nenhum índice válido selecionado. Tente novamente ou deixe vazio para sair.")
                    continue

                # Guarda os outliers manuais para o plot
                indices_para_remover = sorted(list(set(indices_para_remover)), reverse=True) # Evita erros de índice
                tempo_outliers_manual = np.concatenate([tempo_outliers_manual, tempo_minimos_para_fit[indices_para_remover]])
                altura_outliers_manual = np.concatenate([altura_outliers_manual, altura_minimos_para_fit[indices_para_remover]])

                # Aplica o filtro final
                tempo_minimos_para_fit = np.delete(tempo_minimos_para_fit, indices_para_remover)
                altura_minimos_para_fit = np.delete(altura_minimos_para_fit, indices_para_remover)
                print(f"Filtro manual removeu {len(indices_para_remover)} pontos.")
                print(f"Total de {len(altura_minimos_para_fit)} mínimos restantes para a regressão.")
                
                if len(altura_minimos_para_fit) <= grau_polinomio:
                    print("Aviso: Número de pontos é menor ou igual ao grau do polinômio. Finalizando seleção.")
                    break
            # --- FIM DO LOOP 2 ---
    else:
        # Se não for interativo, apenas imprime os pontos usados
        print(f"Total de {len(altura_minimos_para_fit)} mínimos (pós-filtro auto) restantes para a regressão.")

    
    # --- 4. REGRESSÃO POLINOMIAL FINAL ---
    print(f"Calculando regressão final com {len(altura_minimos_para_fit)} pontos e grau {grau_polinomio}.")
    linha_tendencia_polinomial = pd.Series(np.nan, index=tempo.index, dtype=float)

    if len(altura_minimos_para_fit) > grau_polinomio:
        coeficientes = np.polyfit(tempo_minimos_para_fit, altura_minimos_para_fit, grau_polinomio)
        polinomio = np.poly1d(coeficientes)
        tempo_valido_index = tempo.dropna().index
        linha_tendencia_polinomial.loc[tempo_valido_index] = polinomio(tempo.loc[tempo_valido_index])
    else:
        print("Não há pontos suficientes para a regressão final após a filtragem.")
        linha_tendencia_polinomial = pd.Series(np.zeros_like(tempo, dtype=float), index=tempo.index)

    # --- 5. NORMALIZAÇÃO ---
    RFND_Normalizado_Poli = alt - linha_tendencia_polinomial

    # --- 6. VISUALIZAÇÃO FINAL (Se interativo) ---
    if user_interaction and plt is not None:
        plt.figure(figsize=(15, 6))
        plt.plot(tempo, alt, label='Sinal Original (RFND)', alpha=0.5)
        plt.scatter(tempo_outliers_auto, altura_outliers_auto, color='gray', label='Outliers (Automático)', zorder=4)
        if len(tempo_outliers_manual) > 0:
             plt.scatter(tempo_outliers_manual, altura_outliers_manual, marker='x', color='orange', s=100, label='Outliers (Manual)', zorder=5)
        plt.scatter(tempo_minimos_para_fit, altura_minimos_para_fit, color='red', label='Mínimos Finais', zorder=6)
        plt.plot(tempo, linha_tendencia_polinomial, color='black', linestyle='--', linewidth=2, label=f'Tendência Polinomial Final (Grau {grau_polinomio})')
        plt.title('Detecção de Mínimos com Filtragem (Resultado Final)'); plt.xlabel('Tempo (TimeUS)'); plt.ylabel('Altura/Distância'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.show(block=True)

        plt.figure(figsize=(15, 6))
        plt.plot(tempo, RFND_Normalizado_Poli, label='Sinal Normalizado', color='purple')
        plt.axhline(0, color='black', linestyle='--')
        plt.title('Sinal RFND Normalizado (Resultado Final)'); plt.xlabel('Tempo (TimeUS)'); plt.ylabel('Variação de Altura/Distância'); plt.grid(True, alpha=0.3)
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

def align_by_nearest_time_and_interpolate(
    target_time_series: pd.Series,
    data_time_series: pd.Series,
    data_value_series: pd.Series,
    max_time_diff_us: float = 500000  # 0.5 segundos de tolerância
) -> pd.Series:
    """
    (Novo algoritmo de Domínio)
    Alinha uma série de dados (ex: TERR) a uma série de tempo alvo (ex: GPS)
    encontrando o ponto de dados mais próximo no tempo para cada ponto alvo.
    
    1. Para cada timestamp 'alvo', encontra o 'dado' com a menor
       diferença de tempo (se dentro da tolerância).
    2. Cria uma nova série alinhada com muitas lacunas (NaNs).
    3. Preenche as lacunas com uma "transição suave" (interpolação linear).
    4. Preenche as bordas (início/fim) com o valor mais próximo.
    """
    print(f"Alinhando {data_value_series.name} (len={len(data_value_series)}) ao tempo alvo (len={len(target_time_series)})...")
    
    # Converte para arrays numpy para velocidade
    target_times = target_time_series.values
    data_times = data_time_series.values
    data_values = data_value_series.values
    
    # 1. Encontra o índice do tempo de 'data' mais próximo para cada 'target'
    # 'np.searchsorted' é uma forma muito rápida de fazer isso
    # Encontra onde cada 'target_time' se encaixaria no 'data_times'
    indices_insercao = np.searchsorted(data_times, target_times)
    
    # Lida com índices de borda
    indices_insercao[indices_insercao >= len(data_times)] = len(data_times) - 1
    
    # Compara o vizinho da esquerda vs. da direita
    idx_esquerda = indices_insercao - 1
    idx_esquerda[idx_esquerda < 0] = 0 # Garante que não seja < 0
    
    diff_esquerda = np.abs(data_times[idx_esquerda] - target_times)
    diff_direita = np.abs(data_times[indices_insercao] - target_times)
    
    # Escolhe o índice que tiver a menor diferença de tempo
    indices_mais_proximos = np.where(diff_esquerda < diff_direita, idx_esquerda, indices_insercao)
    
    # 2. Cria a nova série alinhada (com lacunas)
    diffs_minimas = np.min(np.vstack([diff_esquerda, diff_direita]), axis=0)
    
    # Inicializa com NaNs
    valores_alinhados = np.full(target_times.shape, np.nan)
    
    # Define o valor apenas se a diferença de tempo estiver dentro da tolerância
    mask_tolerancia = diffs_minimas <= max_time_diff_us
    valores_alinhados[mask_tolerancia] = data_values[indices_mais_proximos[mask_tolerancia]]
    
    # Converte de volta para uma Series Pandas para interpolação
    series_alinhada = pd.Series(valores_alinhados, index=target_time_series.index)
    
    # 3. Preenche as lacunas
    print("Preenchendo lacunas com interpolação linear...")
    # 'linear' faz a "transição suave"
    series_interpolada = series_alinhada.interpolate(method='linear')
    
    # 'ffill' e 'bfill' preenchem as pontas (início/fim) caso
    # a interpolação não consiga
    series_final = series_interpolada.ffill().bfill()
    
    # Quantos valores preenchemos
    nans_originais = series_alinhada.isna().sum()
    print(f"Alinhamento concluído. {nans_originais} lacunas preenchidas.")
    
    return series_final