# domain/shared_algorithms.py
import pandas as pd
import numpy as np
import math
import utm
from typing import Tuple

def calculate_temporal_threshold(gps_time_series: pd.Series) -> float:
    """
    (Lógica movida de LogProcessor.threshold_calc)
    Calcula o threshold de tempo (janela) para o alinhamento dos sensores.
    """
    tempos = gps_time_series.values
    diffs = np.diff(tempos)
    meio_intervalos = diffs / 2
    # Filtra outliers para um threshold mais robusto
    threshold = np.percentile(meio_intervalos, 95)
    return threshold

def calculate_velocity_and_utm(
    tempo: np.ndarray, 
    lat_series: pd.Series, 
    lon_series: pd.Series
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    (Lógica movida de LogProcessor.vel_calc e data_preproxs)
    Converte Lat/Lon para UTM e calcula a velocidade 2D.
    Retorna (velocidade, utm_x, utm_y).
    """
    
    # 1. Converter Lat/Lon para UTM
    x_utm = []
    y_utm = []
    zone_number = None
    zone_letter = None
    
    for lat, lon in zip(lat_series.values, lon_series.values):
        try:
            easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
            x_utm.append(easting)
            y_utm.append(northing)
        except utm.error.OutOfRangeError:
            # Lidar com coordenadas inválidas, se houver
            x_utm.append(np.nan)
            y_utm.append(np.nan)

    x_utm_series = pd.Series(x_utm, index=lat_series.index)
    y_utm_series = pd.Series(y_utm, index=lat_series.index)

    if zone_number:
        print(f"UTM Zone Number: {zone_number}")
        print(f"UTM Zone Letter: {zone_letter}")

    # 2. Calcular Velocidade
    vel = [0]
    tempo_np = tempo
    x_utm_np = x_utm_series.to_numpy()
    y_utm_np = y_utm_series.to_numpy()

    for i in range(len(tempo_np) - 1):
        dt = abs(tempo_np[i+1] - tempo_np[i]) / 1_000_000  # TimeUS to seconds
        dx = abs(x_utm_np[i+1] - x_utm_np[i])
        dy = abs(y_utm_np[i+1] - y_utm_np[i])

        if dt > 0:
            vel_x = dx / dt
            vel_y = dy / dt
            vel.append(math.sqrt(vel_x**2 + vel_y**2))
        else:
            # Adiciona 0 se dt for 0 para evitar divisão por zero
            vel.append(0) 
    
    return pd.Series(vel, index=lat_series.index), x_utm_series, y_utm_series