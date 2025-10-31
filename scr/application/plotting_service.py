# application/plotting_service.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
from scipy import stats
from sklearn.ensemble import IsolationForest

# Importa o "Contrato" da infraestrutura
from scr.domain.interfaces import ILogRepository

class PlottingService:
    """
    Serviço dedicado a lidar com toda a lógica de apresentação (plots).
    Ele lê os arquivos CSV JÁ PROCESSADOS usando o Repositório e os exibe.
    """
    def __init__(self, repo: ILogRepository):
        self.repo = repo
        
        # O "Cardápio" de plots disponíveis
        self.available_plots = {
            "3d_profiles": self.plot_3d_profiles,
            "terr_alt": self.plot_terr_alt_2d,
            "ml_dashboard": self.plot_ml_dashboard,
            "error_comparison": self.plot_error_comparison,
            "error_combined": self.plot_error_combined,
            "error_individual": self.plot_individual_errors,
            "outlier_analysis": self.plot_outlier_analysis,
        }

    def show_available(self) -> None:
        """Mostra o cardápio de plots."""
        print("Uso: /plot <nome_do_plot>")
        print("Plots disponíveis:")
        for key in self.available_plots:
            print(f"- {key}")

    def run_plot(self, plot_name: str) -> None:
        """O 'roteador' que chama a função de plot correta."""
        plot_function = self.available_plots.get(plot_name)
        if plot_function:
            print(f"Gerando plot: '{plot_name}'...")
            try:
                plot_function() # A função de plot carrega seus próprios dados
            except FileNotFoundError as e:
                # Erro pego do repositório se o arquivo não existir
                print(str(e))
            except Exception as e:
                print(f"Erro inesperado ao gerar plot '{plot_name}': {e}")
        else:
            print(f"Erro: Plot '{plot_name}' não encontrado.")
            self.show_available()

    # --- Métodos de Plotagem (Movidos e Adaptados) ---

    def _get_and_clean_plot_data(self) -> pd.DataFrame:
        """(Lógica movida de 2D.graph_plot.py) Helper para carregar e limpar dados_plot2D."""
        data = self.repo.get_processed_dataframe("dados_plot2D.csv")
        
        # Remover linhas com NaN nas colunas importantes
        colunas_importantes = ['TimeMS', 'Alt_BARO', 'Dist','Alt_Ld_BARO','Alt GPS','Alt_Ld_GPS', 'Dist_error', 'TimeuS_max_error']
        data_cleaned = data.dropna(subset=colunas_importantes).copy()
        
        return data_cleaned

    def plot_3d_profiles(self) -> None:
        """(Movido de LogProcessor.plot_data) Gráficos 3D."""
        dados = self.repo.get_processed_dataframe("dados_variados.csv")
        
        x, y = dados['x'], dados['y']
        z_gps, z_baro = dados['Alt Ld GPS'], dados['Alt Ld BARO']
        alt_gps, alt_baro = dados['Alt GPS'], dados['Alt BARO']

        fig1 = plt.figure(figsize=(10, 7))
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.plot(x, y, z_gps, label='Altitude medida (RFND)', color='blue')
        ax1.plot(x, y, alt_gps, label='Altitude (GPS)', color='green')
        ax1.set_title('Perfil 3D: RFND vs GPS'); ax1.set_xlabel('X (UTM)'); ax1.set_ylabel('Y (UTM)'); ax1.set_zlabel('Altitude (m)'); ax1.legend()
        plt.show()

        fig2 = plt.figure(figsize=(10, 7))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot(x, y, z_baro, label='Altitude medida (RFND)', color='blue')
        ax2.plot(x, y, alt_baro, label='Altitude (BARO)', color='purple')
        ax2.set_title('Perfil 3D: RFND vs Barômetro'); ax2.set_xlabel('X (UTM)'); ax2.set_ylabel('Y (UTM)'); ax2.set_zlabel('Altitude (m)'); ax2.legend()
        plt.show()

    def plot_terr_alt_2d(self) -> None:
        """(Movido de LogProcessor.plot_data) Gráfico 2D terr_alt."""
        dados = self.repo.get_processed_dataframe("dados_variados.csv")
        
        tempo_ms = dados['TimeMS']
        terr_alt = dados['terr_alt']
        media_terr_alt = terr_alt.mean()
        print(f"Valor médio (average) de 'terr_alt' é: {media_terr_alt:.2f} m")

        fig3 = plt.figure(figsize=(15, 6))
        ax3 = fig3.add_subplot(111)
        ax3.plot(tempo_ms, terr_alt, label='terr_alt (Alt GPS Norm - Alt BARO)', color='red', alpha=0.8)
        ax3.axhline(media_terr_alt, color='cyan', linestyle='--', label=f'Média ({media_terr_alt:.2f} m)')
        ax3.set_title('Altura do Terreno (terr_alt) vs Tempo'); ax3.set_xlabel('Tempo (ms)'); ax3.set_ylabel('Diferença de Altitude (m)'); ax3.legend(); ax3.grid(True, alpha=0.5)
        plt.show()

    def plot_ml_dashboard(self) -> None:
        """(Movido de LogProcessor.generate_ml_features_and_plot) Dashboard de Features."""
        features_df = self.repo.get_processed_dataframe("features_ml.csv")
        plot_df = features_df.dropna()
        
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 12), sharex=True)
        axes[0].plot(plot_df['TimeMS'], plot_df['z_median'], label='Altitude Mediana (RFND)', color='blue', alpha=0.8)
        axes[0].plot(plot_df['TimeMS'], plot_df['alt_baro'], label='Altitude (BARO)', color='green', linestyle='--', alpha=0.7)
        axes[0].set_ylabel('Altitude (m)'); axes[0].set_title('Dashboard de Features para ML'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(plot_df['TimeMS'], plot_df['z_std_dev'], label='Desvio Padrão (Textura)', color='orange')
        axes[1].set_ylabel('Std Dev (m)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(plot_df['TimeMS'], plot_df['z_amplitude'], label='Amplitude (Max - Min)', color='purple')
        axes[2].set_ylabel('Amplitude (m)'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

        axes[3].plot(plot_df['TimeMS'], plot_df['z_percent_falhas'], label='Percentual de Falhas (Zeros/NaNs)', color='red')
        axes[3].set_ylabel('% Falhas (0.0 a 1.0)'); axes[3].set_xlabel('Tempo (ms)'); axes[3].legend(); axes[3].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def plot_error_comparison(self) -> None:
        """(Movido de 2D.graph_plot.py - método 'plot')"""
        data = self._get_and_clean_plot_data()
        time = data['TimeMS'].values
        rfnd = data['Alt_Ld_GPS'].values # Usando Alt_Ld_GPS como no seu script
        alt = data['Alt GPS'].values
        rfnd_error = data['Dist_error'].values
        
        rfnd_min = rfnd - rfnd_error
        rfnd_max = rfnd + rfnd_error

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time, rfnd, color='blue', linewidth=2, label='RFND (Ld_GPS)')
        ax.plot(time, alt, color='green', linewidth=2, label='Altitude (GPS)')
        ax.fill_between(time, rfnd_min, rfnd_max, color='blue', alpha=0.3, label='Erro RFND (±σ)')
        
        ax.set_title('Comparação entre GPS e RFND com Erro'); ax.set_xlabel('Tempo (ms)'); ax.set_ylabel('Altura/Distância'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.show()

    def plot_error_combined(self) -> None:
        """(Movido de 2D.graph_plot.py - método 'plot_combined_error')"""
        data = self._get_and_clean_plot_data()
        time = data['TimeMS'].values
        rfnd = data['Alt_Ld_GPS'].values
        alt = data['Alt GPS'].values
        rfnd_error = data['Dist_error'].values
        time_error = data['TimeuS_max_error'].values / 1000.0 # ms
        
        rfnd_min, rfnd_max = rfnd - rfnd_error, rfnd + rfnd_error
        time_min, time_max = time - time_error, time + time_error
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(len(time)):
            vertices = [[time_min[i], rfnd_min[i]], [time_max[i], rfnd_min[i]], [time_max[i], rfnd_max[i]], [time_min[i], rfnd_max[i]]]
            polygon = Polygon(np.array(vertices), closed=True, color='purple', alpha=0.2, edgecolor='purple', linewidth=0.5)
            ax.add_patch(polygon)
        
        ax.plot(time, rfnd, 'b-', linewidth=2, label='RFND (Ld_GPS)')
        ax.plot(time, alt, 'g-', linewidth=2, label='GPS')
        ax.set_title('RFND com Área de Erro Combinada (Temporal + Amplitude)'); ax.set_xlabel('Tempo (ms)'); ax.set_ylabel('Altura/Distância'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.show()
        
    def plot_individual_errors(self) -> None:
        """(Movido de 2D.graph_plot.py - método 'plot_individual')"""
        data = self._get_and_clean_plot_data()
        time = data['TimeMS'].values
        rfnd = data['Alt_Ld_GPS'].values
        alt = data['Alt GPS'].values
        rfnd_error = data['Dist_error'].values
        time_error = data['TimeuS_max_error'].values / 1000.0 # ms
        rfnd_min, rfnd_max = rfnd - rfnd_error, rfnd + rfnd_error
        time_min, time_max = time - time_error, time + time_error

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        
        for i in range(len(time)): # Erro combinado
            vertices = [[time_min[i], rfnd_min[i]], [time_max[i], rfnd_min[i]], [time_max[i], rfnd_max[i]], [time_min[i], rfnd_max[i]]]
            polygon = Polygon(np.array(vertices), closed=True, color='purple', alpha=0.2, edgecolor='purple', linewidth=0.5)
            axes[0, 0].add_patch(polygon)
        axes[0, 0].plot(time, rfnd, 'b-', linewidth=2, label='RFND'); axes[0, 0].set_title('RFND com Erro Combinado'); axes[0, 0].set_ylabel('Valor'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].errorbar(time, alt, xerr=time_error, fmt='go-', ecolor='orange', alpha=0.7, capsize=3, label='GPS com erro temporal'); axes[0, 1].set_title('Altura GPS com Erro Temporal'); axes[0, 1].set_ylabel('Altura'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(time, time_error, 'orange', label='Erro Temporal (ms)', linewidth=2); axes[1, 0].set_title('Componentes do Erro'); axes[1, 0].set_xlabel('Tempo (ms)'); axes[1, 0].set_ylabel('Magnitude do Erro'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time, rfnd_error, 'r-', label='Erro de Amplitude (m)', linewidth=2); axes[1, 1].set_title('Componentes do Erro'); axes[1, 1].set_xlabel('Tempo (ms)'); axes[1, 1].set_ylabel('Magnitude do Erro'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_outlier_analysis(self) -> None:
        """(Movido de 2D.graph_plot.py - método 'executar_analise_visual_completa')"""
        data = self._get_and_clean_plot_data()
        time = data['TimeMS'].values
        rfnd = data['Alt_Ld_GPS'].values
        
        dados_serie = pd.Series(rfnd)
        df_analise = pd.DataFrame({'tempo': time, 'valor': dados_serie})

        # IQR
        Q1, Q3 = dados_serie.quantile(0.25), dados_serie.quantile(0.75)
        IQR = Q3 - Q1
        lim_inf_iqr, lim_sup_iqr = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask_iqr = (dados_serie < lim_inf_iqr) | (dados_serie > lim_sup_iqr)

        # Z-score
        limiar_z = 2.0
        mask_zscore = np.abs(stats.zscore(dados_serie)) > limiar_z
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination='auto', random_state=42)
        mask_iso = iso_forest.fit_predict(df_analise[['valor']]) == -1

        # Plot 1: Comparação de Métodos
        fig1, axes1 = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
        fig1.suptitle('Comparação de Métodos de Detecção de Outliers vs. Tempo', fontsize=16)
        axes1[0].scatter(df_analise['tempo'], df_analise['valor'], c='blue', label='Normais', alpha=0.7)
        axes1[0].scatter(df_analise['tempo'][mask_iqr], df_analise['valor'][mask_iqr], c='red', marker='x', s=100, label='Outliers')
        axes1[0].axhline(y=lim_sup_iqr, color='gray', linestyle='--'); axes1[0].axhline(y=lim_inf_iqr, color='gray', linestyle='--'); axes1[0].set_title('Método IQR'); axes1[0].set_xlabel('Tempo (ms)'); axes1[0].set_ylabel('Valor (RFND)'); axes1[0].legend()
        
        axes1[1].scatter(df_analise['tempo'], df_analise['valor'], c='blue', label='Normais', alpha=0.7)
        axes1[1].scatter(df_analise['tempo'][mask_zscore], df_analise['valor'][mask_zscore], c='red', marker='x', s=100, label='Outliers'); axes1[1].set_title(f'Método Z-score (Limiar = {limiar_z})'); axes1[1].set_xlabel('Tempo (ms)'); axes1[1].legend()
        
        axes1[2].scatter(df_analise['tempo'], df_analise['valor'], c='blue', label='Normais', alpha=0.7)
        axes1[2].scatter(df_analise['tempo'][mask_iso], df_analise['valor'][mask_iso], c='red', marker='x', s=100, label='Outliers'); axes1[2].set_title('Método Isolation Forest'); axes1[2].set_xlabel('Tempo (ms)'); axes1[2].legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

        # Plot 2: Progressão Temporal Filtrada
        fig2, axes2 = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig2.suptitle('Progressão Temporal do Sinal RFND Filtrado', fontsize=16)
        axes2[0].plot(df_analise['tempo'][~mask_iqr], df_analise['valor'][~mask_iqr], color='blue', label='RFND Filtrado'); axes2[0].scatter(df_analise['tempo'][mask_iqr], df_analise['valor'][mask_iqr], marker='x', color='red', label='Outliers Removidos'); axes2[0].set_title('Filtro IQR'); axes2[0].set_ylabel('Altitude (m)'); axes2[0].legend(); axes2[0].grid(True, linestyle='--', linewidth=0.5)
        axes2[1].plot(df_analise['tempo'][~mask_zscore], df_analise['valor'][~mask_zscore], color='green', label='RFND Filtrado'); axes2[1].scatter(df_analise['tempo'][mask_zscore], df_analise['valor'][mask_zscore], marker='x', color='red', label='Outliers Removidos'); axes2[1].set_title('Filtro Z-score'); axes2[1].set_ylabel('Altitude (m)'); axes2[1].legend(); axes2[1].grid(True, linestyle='--', linewidth=0.5)
        axes2[2].plot(df_analise['tempo'][~mask_iso], df_analise['valor'][~mask_iso], color='purple', label='RFND Filtrado'); axes2[2].scatter(df_analise['tempo'][mask_iso], df_analise['valor'][mask_iso], marker='x', color='red', label='Outliers Removidos'); axes2[2].set_title('Filtro Isolation Forest'); axes2[2].set_xlabel('Tempo (ms)'); axes2[2].set_ylabel('Altitude (m)'); axes2[2].legend(); axes2[2].grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()