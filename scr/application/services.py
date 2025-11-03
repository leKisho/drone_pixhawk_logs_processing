# application/services.py
import pandas as pd
import numpy as np
from typing import List, Optional

from scr.domain.interfaces import ILogRepository
from scr.domain import sensor_processing
from scr.domain import feature_engineering
from scr.domain import shared_algorithms
from scr.infrastructure.log_parser import LogLabeler, LogSeparator
from scr.application.plotting_service import PlottingService

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
                            modo_interativo: bool = True,
                            grau_polinomio: int = 8) -> None:
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
                Tempo_GPS, z_gps, 
                user_interaction=modo_interativo,
                initial_degree=grau_polinomio  # <-- PASSE O GRAU AQUI
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


    def _get_threshold_input(self, prompt: str) -> float:
        """Pede ao usuário um valor de threshold e valida."""
        while True:
            try:
                value_str = input(prompt)
                return float(value_str)
            except ValueError:
                print("Entrada inválida. Por favor, digite um número (ex: 0.5).")


    # --- 3. MODIFIQUE O CASO DE USO ---
    def generate_programmatic_labels(self, plotter: PlottingService, labeling_source: str = "barometer") -> Optional[pd.DataFrame]:
        """
        Caso de Uso: /label
        Roda o assistente interativo para criar o gabarito de ML.
        Usa a fonte de altitude (barometer ou gps_corrigido)
        especificada pela flag no 'main.py'.
        """
        
        # --- PASSO 1: Carregar os dados necessários ---
        try:
            # O 'features_ml' (tabela) tem as FEATURES (z_std_dev, etc.)
            df_features = self.repo.get_processed_dataframe("features_ml.csv")
            
            # O 'dados_plot2D' (tabela) tem as ALTITUDES (os alvos)
            # Ele já contém 'Alt_Ld_GPS' (GPS corrigido) E 'Alt_Ld_BARO' (Barômetro corrigido)
            df_altitude = self.repo.get_processed_dataframe("dados_plot2D.csv")
            
        except FileNotFoundError as e:
            print(f"Erro ao carregar dados: {e}")
            print("\nCertifique-se de que você já executou o pipeline principal ('python main.py')")
            print("pois ele é necessário para gerar as tabelas 'features_ml' e 'dados_plot2D' primeiro.")
            return None
            
        print(f"Carregadas {len(df_features)} features e {len(df_altitude)} pontos de altitude.")

        # --- PASSO 2: Unir os dados ---
        # (O gabarito final precisa das features E do alvo)
        df_features['TimeMS_join'] = df_features['TimeMS'].round(3)
        df_altitude['TimeMS_join'] = df_altitude['TimeMS'].round(3)
        
        df_gabarito = pd.merge(
            df_features, 
            df_altitude[['TimeMS_join', 'Alt_Ld_GPS', 'Alt_Ld_BARO']], # Pega AMBAS as altitudes
            on='TimeMS_join',
            how='inner'
        )
        
        if df_gabarito.empty:
            print("Erro: Não foi possível unir 'features_ml' e 'dados_plot2D'. Verifique os dados de TimeMS.")
            return None
        
        print(f"Junção resultou em {len(df_gabarito)} amostras válidas.")

        # --- PASSO 3: Rotular pela Altitude (DINÂMICO) ---
        
        # Escolhe qual coluna usar com base na flag do main.py
        if labeling_source.lower() == "barometer":
            altitude_col_name = 'Alt_Ld_BARO'
            plot_title = "PASSO ÚNICO: Decida o threshold de ALTITUDE (Barômetro)"
        else:
            altitude_col_name = 'Alt_Ld_GPS'
            plot_title = "PASSO ÚNICO: Decida o threshold de ALTITUDE (GPS Corrigido)"

        print(f"\n--- Usando a fonte de altitude: '{altitude_col_name}' ---")
        
        # Delegue o PLOT para o PlottingService
        plotter._show_labeling_histogram(
            df_gabarito[altitude_col_name], # <-- Usa a coluna dinâmica
            plot_title,
            f"Altitude da Superfície ({altitude_col_name}) [m]"
        )
        
        # O ApplicationService cuida da lógica (input)
        print(f"Baseado no gráfico de '{altitude_col_name}', decida qual valor de ALTITUDE")
        print("separa o 'Solo/Água' (baixa altitude) da 'Vegetação' (alta altitude).")
        threshold_altitude = self._get_threshold_input(f"Threshold de Altitude (ex: 3.5): ")

        # --- Aplicar as regras para criar as classes ---
        print("\nAplicando regras...")
        print(f"  Regra 1 (Vegetação - Classe 1): {altitude_col_name} >= {threshold_altitude}")
        print(f"  Regra 2 (Solo/Água - Classe 0): {altitude_col_name} < {threshold_altitude}")
        
        df_gabarito['Classe'] = 0
        df_gabarito.loc[df_gabarito[altitude_col_name] >= threshold_altitude, 'Classe'] = 1
        
        mask_nan = df_gabarito[altitude_col_name].isna() | df_gabarito['z_std_dev'].isna()
        df_gabarito.loc[mask_nan, 'Classe'] = np.nan

        print("Geração de labels (baseada em altitude) concluída.")
        
        # Retorna o DataFrame completo (com features e labels) para o 'main' salvar
        # O MLService usará as 'features_ml' para prever a 'Classe'
        return df_gabarito
    
    def process_main_data(self, mins: int = 0, maxs: int = -1, 
                            aplicar_correcao: bool = True, 
                            modo_interativo: bool = True,
                            grau_polinomio: int = 8) -> None:
        """
        Caso de Uso: /data
        Orquestra o fluxo de processamento de dados completo.
        """
        print("Iniciando processamento principal de dados (/data)...")
        
        # --- 1. Obter Dados (Infraestrutura) ---
        # 1A. MODIFICAÇÃO: Adicione "TERR" à lista de sensores necessários
        required_sensors = ("GPS", "RFND", "BARO", "TERR")
        
        print(f"Requisitando sensores do repositório: {required_sensors}")
        raw_data = self.repo.get_raw_sensor_data(required_sensors)
        
        # Validação
        if not all(k in raw_data for k in ("GPS", "RFND", "BARO")):
            print("Erro: Faltam dados essenciais (GPS, RFND, BARO). Abortando.")
            return
            
        gps_df = raw_data["GPS"]
        rfnd_df = raw_data["RFND"]
        baro_df = raw_data["BARO"]
        
        # 1B. MODIFICAÇÃO: Lida com o 'TERR' (pode não existir em logs antigos)
        terr_df = raw_data.get("TERR") # Retorna None se não existir
        if terr_df is not None:
            print(f"Dados TERR (height) encontrados ({len(terr_df)} pontos).")
            # Remove linhas onde TerrH é NaN (se houver)
            terr_df = terr_df.dropna(subset=['TimeUS', 'TerrH'])
        else:
            print("Aviso: Dados TERR (height) não encontrados no log. Altitude do solo não será calculada.")

        # --- 2. Fatiar Dados (slicing) ---
        # (Sem mudanças aqui)
        gps_df_sliced = gps_df.iloc[mins:maxs].copy()
        Tempo_GPS = gps_df_sliced['TimeUS']
        Alt_GPS = gps_df_sliced['Alt']
        y_lat = gps_df_sliced['Lat']
        x_lon = gps_df_sliced['Lng']
            
        # --- 3. Chamar Algoritmos do Domínio ---
        print("Executando algoritmos do domínio...")
        
        # 3a. Calcular Threshold (Sem mudanças)
        threshold = shared_algorithms.calculate_temporal_threshold(gps_df['TimeUS'])
        
        # 3b. Alinhar Sensores RFND/BARO (Sem mudanças)
        (z_agrupado, erros_tempo, 
         amplitude_z, alt_agrupada) = sensor_processing.align_sensor_data(
            gps_df, rfnd_df, baro_df, threshold
        )
        
        # 3c. Fatiar os resultados alinhados (Sem mudanças)
        alt_baro_sliced = alt_agrupada.iloc[mins:maxs]
        z_sliced = z_agrupado.iloc[mins:maxs]
        erros_tempo_sliced = erros_tempo[mins:maxs]
        amplitude_z_sliced = amplitude_z[mins:maxs]

        # 3d. MODIFICAÇÃO: Alinhar e Interpolar o TERR
        if terr_df is not None and not terr_df.empty:
            alt_solo_terr = sensor_processing.align_by_nearest_time_and_interpolate(
                target_time_series=Tempo_GPS,
                data_time_series=terr_df['TimeUS'],
                data_value_series=terr_df['TerrH'].rename("Alt_Solo_TERR") # Dê um nome
            )
        else:
            print("Pulando cálculo de Alt_Solo_TERR.")
            alt_solo_terr = pd.Series(np.nan, index=Tempo_GPS.index, name="Alt_Solo_TERR")

        # 3e. Cálculos restantes (Vel, UTM, correções de GPS/BARO)
        # (Sem mudanças aqui, mantemos as correções originais caso sejam úteis)
        vel, x_utm, y_utm = shared_algorithms.calculate_velocity_and_utm(Tempo_GPS.values, y_lat, x_lon)
        z_gps = Alt_GPS - z_sliced
        z_baro = alt_baro_sliced - z_sliced

        if aplicar_correcao:
            # ... (bloco da correção polinomial, sem mudanças) ...
            z_gps_normalizado, tendencia_z_gps = sensor_processing.normalize_signal_with_polynomial_fit(
                Tempo_GPS, z_gps, user_interaction=modo_interativo, initial_degree=grau_polinomio
            )
        else:
            # ... (bloco else, sem mudanças) ...
            z_gps_normalizado = z_gps
            tendencia_z_gps = pd.Series(0.0, index=Tempo_GPS.index)
        
        alt_gps_normalizado = Alt_GPS - tendencia_z_gps
        z_gps_final = z_gps_normalizado + Alt_GPS.iloc[0] 
        terr_alt = alt_gps_normalizado - alt_baro_sliced # (Esta é a 'terr_alt' antiga)
        z_baro_filtrado = sensor_processing.filter_outliers_zscore(z_baro)
        Tempo_GPSms = (Tempo_GPS - Tempo_GPS.iloc[0]) / 1000.0

        # --- 4. Salvar Resultados (Infraestrutura) ---
        print("Salvando DataFrames processados...")
        
        # 4a. dados_variados.csv
        # MODIFICAÇÃO: Adicione a nova coluna 'Alt_Solo_TERR'
        df_variados = pd.DataFrame({
            'TimeUS': Tempo_GPS, 'TimeUS_max_error': erros_tempo_sliced,
            'TimeMS': Tempo_GPSms, 'Dist': z_sliced, 'Lon': x_lon, 'Lat': y_lat,
            'x': x_utm, 'y': y_utm, 'Alt GPS': Alt_GPS, 'Alt BARO': alt_baro_sliced,
            'Alt Ld GPS': z_gps_final, 'Alt Ld BARO': z_baro_filtrado, 
            'Alt_Solo_TERR': alt_solo_terr, # <-- SUA NOVA COLUNA AQUI
            'terr_alt': terr_alt, 'Dist_error': amplitude_z_sliced,
            'Vel': vel
        }).dropna(subset=['TimeUS']) # (Só drope se TimeUS for NaN, se houver)
        
        self.repo.save_processed_dataframe(df_variados, "dados_variados.csv")

        # 4b. dados_plot2D.csv
        # MODIFICAÇÃO: Adicione a nova coluna aqui também
        df_plot2d = pd.DataFrame({
            'TimeMS': Tempo_GPSms, 'TimeuS_max_error': erros_tempo_sliced,
            'Dist': z_sliced, 'Alt_BARO': alt_baro_sliced, 'Alt GPS': Alt_GPS,
            'Alt_Ld_BARO': z_baro, 'Alt_Ld_GPS': z_gps_final, 
            'Alt_Solo_TERR': alt_solo_terr, # <-- SUA NOVA COLUNA AQUI
            'terr_alt': terr_alt, 'Dist_error': amplitude_z_sliced,
        }).dropna(subset=['TimeMS'])
        
        self.repo.save_processed_dataframe(df_plot2d, "dados_plot2D.csv")

        # 4c. dados_interp.csv
        # MODIFICAÇÃO: Adicione a nova coluna aqui também
        df_interp = pd.DataFrame({
            'TimeUS': Tempo_GPS, 'x': x_utm, 'Y': y_utm,
            'Alt_Ld_Baro': z_baro_filtrado, 
            'Alt_Solo_TERR': alt_solo_terr, # <-- SUA NOVA COLUNA AQUI
            'terr_alt': terr_alt
        }).dropna()
        self.repo.save_processed_dataframe(df_interp, "dados_interp.csv")
        
        print("Processamento principal de dados concluído.")