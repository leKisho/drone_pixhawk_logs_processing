# domain/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict

def generate_lidar_features(
    gps_df: pd.DataFrame, 
    rfnd_df: pd.DataFrame, 
    baro_df: pd.DataFrame, 
    threshold: float
) -> pd.DataFrame:
    """
    (Lógica Pura movida de LogProcessor.z_filter_for_ml)
    Aplica o janelamento temporal e calcula as features de ML (StdDev, Falhas, etc.)
    Retorna um DataFrame completo alinhado com o tempo do GPS.
    """
    print("Iniciando extração de features para ML...")
    
    # 1. Prepara dados do RFND
    rfnd_clean = rfnd_df.copy()
    if 'Stat1' in rfnd_clean.columns:
        mask = rfnd_clean['Stat1'] != 4
        rfnd_clean.loc[mask, 'Dist1'] = np.nan

    # 2. Prepara arrays para loop
    tempos_gps = gps_df['TimeUS'].values
    tempos_z = rfnd_clean['TimeUS'].values
    tempos_baro = baro_df['TimeUS'].values

    valores_z = rfnd_clean['Dist1'].values
    valores_alt = baro_df['Alt'].values

    # 4. Listas para armazenar as features calculadas
    feature_z_median = []
    feature_z_std_dev = []
    feature_z_amplitude = []
    feature_z_percent_falhas = []
    feature_alt_baro = []

    # 5. Loop principal (O Janelamento)
    for t in tempos_gps:
        z_local_na_janela = []
        alt_local_na_janela = []

        for tz, z in zip(tempos_z, valores_z):
            if abs(tz-t) <= threshold:
                z_local_na_janela.append(z)
        
        for ta, a in zip(tempos_baro, valores_alt):
            if abs(ta-t) <= threshold:
                alt_local_na_janela.append(a)
        
        # --- Extração das Features ---
        total_leituras_z = len(z_local_na_janela)
        
        # Filtra NaNs (Stat1!=4) e Zeros (falha de leitura)
        z_validos = [v for v in z_local_na_janela if pd.notna(v) and v > 0]
        contagem_validos = len(z_validos)

        # Feature 1: Percentual de Falhas (A "Água")
        if total_leituras_z > 0:
            falhas = (total_leituras_z - contagem_validos) / total_leituras_z
            feature_z_percent_falhas.append(falhas)
        else:
            feature_z_percent_falhas.append(1.0) # 100% falha

        # Features 2, 3, 4 (calculadas sobre os dados válidos)
        if contagem_validos > 0:
            feature_z_median.append(np.median(z_validos))
            feature_z_std_dev.append(np.std(z_validos))
            feature_z_amplitude.append(np.max(z_validos) - np.min(z_validos))
        else:
            feature_z_median.append(np.nan)
            feature_z_std_dev.append(np.nan)
            feature_z_amplitude.append(np.nan)
        
        # Feature do Barômetro
        if alt_local_na_janela:
            feature_alt_baro.append(np.median(alt_local_na_janela))
        else:
            feature_alt_baro.append(np.nan)

    print("Extração de features concluída.")
    
    # 6. Monta e retorna o DataFrame
    features_df = pd.DataFrame({
        'TimeUS': tempos_gps,
        'TimeMS': (tempos_gps - tempos_gps[0]) / 1000.0,
        'z_median': feature_z_median,
        'z_std_dev': feature_z_std_dev,
        'z_amplitude': feature_z_amplitude,
        'z_percent_falhas': feature_z_percent_falhas,
        'alt_baro': feature_alt_baro
    })
    
    return features_df