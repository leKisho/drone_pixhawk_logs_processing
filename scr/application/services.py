# application/services.py
import pandas as pd
from typing import List, Optional

from scr.domain.interfaces import ILogRepository
from scr.domain import sensor_processing
from scr.domain import feature_engineering
from scr.domain import shared_algorithms
from scr.infrastructure.log_parser import LogLabeler, LogSeparator

class ApplicationService:
    """
    Orquestra os casos de uso da aplicação.
    Ele atua como o "Gerente" que coordena o Domínio e a Infraestrutura.
    """
    def __init__(self, repo: ILogRepository, log_filepath: str):
        self.repo = repo
        self.log_filepath = log_filepath
        self.labels = None
        self.ids = None

    def _get_labels(self) -> List:
        """Helper interno para carregar os labels (IDs) do log."""
        if self.labels is None:
            # Passa a função de extração para o repositório, que
            # decidirá se a executa ou se usa o cache (.npz).
            labeler = LogLabeler(self.log_filepath)
            self.ids, self.labels = self.repo.load_or_create_labels(
                extractor_func=labeler.extract_labels
            )
        return self.labels

    def show_available_ids(self) -> None:
        """
        Caso de Uso: /show
        (Substitui o LogProcessor.show_ids)
        """
        self._get_labels()
        t = 1
        print("IDs disponíveis no arquivo de log:")
        for ident in sorted(pd.unique([item for sublist in self.ids for item in sublist])):
            print(ident, end="; ")
            if t % 10 == 0:
                print()
            t += 1
        print()

    def process_logs_to_csv(self, id_list: Optional[List[str]] = None) -> None:
        """
        Caso de Uso: /all OU "ahr2,baro,..."
        (Substitui LogProcessor.process_all e process_selected)
        Extrai os dados brutos do .log e salva em CSVs individuais.
        """
        labels = self._get_labels()
        
        if id_list:
            print(f"Processando IDs selecionados: {', '.join(id_list)}")
            id_list_upper = [name.strip().upper() for name in id_list]
        else:
            print("Processando todos os IDs (/all)...")
            id_list_upper = None
            
        data_subdir = self.repo.data_subdir # Pega o caminho de saída do repositório
            
        for label in labels:
            name_clean = str(label[0]).strip("[]'")
            if id_list_upper is None or name_clean in id_list_upper:
                # Delega a tarefa de separar o log para a classe da infraestrutura
                LogSeparator(label, self.log_filepath, data_subdir).save_to_csv()
        
        print("Separação dos logs em CSVs concluída.")

    def process_main_data(self, mins: int = 0, maxs: int = -1, 
                            aplicar_correcao: bool = True, 
                            modo_interativo: bool = True) -> None:
        """
        Caso de Uso: /data
        (Substitui LogProcessor.data_preproxs)
        Orquestra o fluxo de processamento de dados completo.
        """
        print("Iniciando processamento principal de dados (/data)...")
        
        # 1. Obter Dados (Infraestrutura)
        # ... (código de obter dados, gps_df, rfnd_df, baro_df) ...
        raw_data = self.repo.get_raw_sensor_data(("GPS", "RFND", "BARO"))
        gps_df = raw_data["GPS"]
        rfnd_df = raw_data["RFND"]
        baro_df = raw_data["BARO"]


        # 2. Fatiar Dados (slicing)
        # ... (código de slicing, Tempo_GPS, Alt_GPS, etc.) ...
        gps_df_sliced = gps_df.iloc[mins:maxs].copy()
        Tempo_GPS = gps_df_sliced['TimeUS']
        Alt_GPS = gps_df_sliced['Alt']
        y_lat = gps_df_sliced['Lat']
        x_lon = gps_df_sliced['Lng']
            
        # 3. Chamar Algoritmos do Domínio
        print("Executando algoritmos do domínio...")
        
        # 3a. Calcular Threshold
        threshold = shared_algorithms.calculate_temporal_threshold(gps_df['TimeUS'])
        
        # 3b. Alinhar Sensores (lógica do z_filter)
        # <--- ATUALIZADO: tupla de retorno agora tem 4 itens ---
        (z_agrupado, erros_tempo, 
         amplitude_z, alt_agrupada) = sensor_processing.align_sensor_data(
            gps_df, rfnd_df, baro_df, threshold
        )
        
        # 3c. Fatiar os resultados alinhados
        alt_baro_sliced = alt_agrupada.iloc[mins:maxs]
        z_sliced = z_agrupado.iloc[mins:maxs]
        erros_tempo_sliced = erros_tempo[mins:maxs]
        amplitude_z_sliced = amplitude_z[mins:maxs]

        # ... (3d a 3h, vel, utm, normalize, zscore... tudo igual) ...
        vel, x_utm, y_utm = shared_algorithms.calculate_velocity_and_utm(Tempo_GPS.values, y_lat, x_lon)
        z_gps = Alt_GPS - z_sliced
        z_baro = alt_baro_sliced - z_sliced

        if aplicar_correcao:
            print(f"Executando correção polinomial (Modo interativo: {modo_interativo})...")
            z_gps_normalizado, tendencia_z_gps = sensor_processing.normalize_signal_with_polynomial_fit(
                Tempo_GPS, z_gps, user_interaction=modo_interativo
            )
        else:
            print("Correção polinomial PULADA. Usando 'z_gps' original.")
            z_gps_normalizado = z_gps
            # Define a tendência como 0 para não afetar os cálculos seguintes
            tendencia_z_gps = pd.Series(0.0, index=Tempo_GPS.index)


        alt_gps_normalizado = Alt_GPS - tendencia_z_gps
        z_gps_final = z_gps_normalizado + Alt_GPS.iloc[0] 
        terr_alt = alt_gps_normalizado - alt_baro_sliced
        z_baro_filtrado = sensor_processing.filter_outliers_zscore(z_baro)
        Tempo_GPSms = (Tempo_GPS - Tempo_GPS.iloc[0]) / 1000.0

        # 4. Salvar Resultados (Infraestrutura)
        print("Salvando DataFrames processados...")
        
        # 4a. dados_variados.csv
        df_variados = pd.DataFrame({
            'TimeUS': Tempo_GPS, 'TimeUS_max_error': erros_tempo_sliced,
            'TimeMS': Tempo_GPSms, 'Dist': z_sliced, 'Lon': x_lon, 'Lat': y_lat,
            'x': x_utm, 'y': y_utm, 'Alt GPS': Alt_GPS, 'Alt BARO': alt_baro_sliced,
            'Alt Ld GPS': z_gps_final, 'Alt Ld BARO': z_baro_filtrado, 
            'terr_alt': terr_alt, 'Dist_error': amplitude_z_sliced,
            # 'Dist_ls': z_ls_sliced, <--- REMOVIDO
            'Vel': vel
        }).dropna()
        self.repo.save_processed_dataframe(df_variados, "dados_variados.csv")

        # 4b. dados_plot2D.csv
        df_plot2d = pd.DataFrame({
            'TimeMS': Tempo_GPSms, 'TimeuS_max_error': erros_tempo_sliced,
            'Dist': z_sliced, 'Alt_BARO': alt_baro_sliced, 'Alt GPS': Alt_GPS,
            'Alt_Ld_BARO': z_baro, 'Alt_Ld_GPS': z_gps_final, 'terr_alt': terr_alt, 
            'Dist_error': amplitude_z_sliced,
            # 'Dist_ls': z_ls_sliced <--- REMOVIDO
        }).dropna()
        self.repo.save_processed_dataframe(df_plot2d, "dados_plot2D.csv")

        # 4c. dados_interp.csv
        # ... (sem mudanças aqui, já não usava z_ls) ...
        df_interp = pd.DataFrame({
            'TimeUS': Tempo_GPS, 'x': x_utm, 'Y': y_utm,
            'Alt_Ld_Baro': z_baro_filtrado, 'terr_alt': terr_alt
        }).dropna()
        self.repo.save_processed_dataframe(df_interp, "dados_interp.csv")
        
        print("Processamento principal de dados concluído.")

    def generate_ml_features(self) -> None:
        """
        Caso de Uso: /features
        (Substitui LogProcessor.generate_ml_features_and_plot - parte lógica)
        Orquestra a extração de features para ML.
        """
        print("Iniciando geração de features para ML (/features)...")
        
        # 1. Obter Dados (Infraestrutura)
        print("Carregando dados brutos (GPS, RFND, BARO)...")
        raw_data = self.repo.get_raw_sensor_data(("GPS", "RFND", "BARO"))
        if not all(k in raw_data for k in ("GPS", "RFND", "BARO")):
            print("Erro: Faltam dados essenciais. Execute a separação primeiro.")
            return

        gps_df = raw_data["GPS"]
        rfnd_df = raw_data["RFND"]
        baro_df = raw_data["BARO"]
        
        # 2. Chamar Algoritmos do Domínio
        print("Executando algoritmos do domínio...")
        
        # 2a. Calcular Threshold
        threshold = shared_algorithms.calculate_temporal_threshold(gps_df['TimeUS'])

        # 2b. Gerar Features (lógica do z_filter_for_ml)
        features_df = feature_engineering.generate_lidar_features(
            gps_df, rfnd_df, baro_df, threshold
        )
        
        # 3. Salvar Resultados (Infraestrutura)
        print("Salvando DataFrame de features...")
        self.repo.save_processed_dataframe(features_df, "features_ml.csv")
        
        print("Geração de features concluída.")