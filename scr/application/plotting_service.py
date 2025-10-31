#
# Arquivo: scr/application/plotting_service.py
#
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict

# Importa a Interface (Contrato)
from scr.domain.interfaces import ILogRepository

class PlottingService:
    """
    Serviço de Aplicação focado em criar e salvar visualizações.
    """
    
    def __init__(self, repository: ILogRepository):
        self.repo = repository
        # O 'log_dir' ainda é útil para logs de debug ou outros
        self.log_dir = repository.log_dir 
        
        # --- MUDANÇA (Missão 1) ---
        # Pega o caminho do diretório raiz do projeto (subindo 3 níveis: .../scr/application/plotting_service.py -> .../scr/application -> .../scr -> ROOT)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Define o novo diretório de plots
        self.plot_dir = os.path.join(base_dir, "assets", "plotts") 
        # --- FIM DA MUDANÇA ---
        
        # Garante que o diretório de destino exista
        os.makedirs(self.plot_dir, exist_ok=True)
        print(f"Plots serão salvos em: {self.plot_dir}")

        # Configura o Plotly para usar o Kaleido para exportar imagens
        pio.kaleido.scope.mathjax = None

    def run_plot(self, plot_name: str):
        """
        Executa uma rotina de plotagem específica.
        """
        if plot_name == "3d_profiles":
            self._plot_3d_profiles()
        elif plot_name == "terr_alt":
            self._plot_terr_alt()
        elif plot_name == "ml_dashboard":
            self._plot_ml_dashboard()
        elif plot_name == "outlier_analysis":
            self._plot_outlier_analysis()
        else:
            print(f"Aviso: Plot '{plot_name}' desconhecido.")

    # ... (O restante do arquivo continua exatamente igual) ...
    
    def _plot_terr_alt(self):
        """
        PLANO B (Matplotlib): Plota Altitude GPS vs Altitude Terreno (LIDAR)
        """
        print("Executando plot 'terr_alt' (Matplotlib)...")
        try:
            df = self.repo.get_processed_dataframe("dados_variados.csv")
        except FileNotFoundError as e:
            print(f"Erro ao plotar: {e}. Pule esta etapa.")
            return

        plt.figure(figsize=(15, 7))
        plt.plot(df['Time_s'], df['Alt_m'], label='Altitude (GPS)', color='blue')
        plt.plot(df['Time_s'], df['TerrAlt_m'], label='Altitude do Terreno (LIDAR)', color='orange')
        plt.plot(df['Time_s'], df['RelAlt_m'], label='Altitude Relativa (GPS - LIDAR)', color='green', linestyle='--')
        
        plt.title('Perfil de Voo e Detecção do Terreno')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Altitude (m)')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.plot_dir, f'plot_terr_alt_{self.repo.log_id}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Plot 'terr_alt' salvo em: {save_path}")

    def _plot_3d_profiles(self):
        """
        PLANO B (Matplotlib): Plota perfis 3D (Posição e Altitude)
        """
        print("Executando plot '3d_profiles' (Matplotlib)...")
        try:
            df = self.repo.get_processed_dataframe("dados_variados.csv")
        except FileNotFoundError as e:
            print(f"Erro ao plotar: {e}. Pule esta etapa.")
            return

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot 3D da trajetória
        ax.plot(df['Lon_deg'], df['Lat_deg'], df['Alt_m'], label='Trajetória 3D (GPS)')
        
        # Projeção 2D no "chão" (altitude 0)
        ax.plot(df['Lon_deg'], df['Lat_deg'], 0, label='Projeção 2D (Chão)', color='grey', linestyle='--')

        ax.set_xlabel('Longitude (deg)')
        ax.set_ylabel('Latitude (deg)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('Trajetória de Voo 3D')
        ax.legend()
        
        save_path = os.path.join(self.plot_dir, f'plot_3d_profiles_{self.repo.log_id}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Plot '3d_profiles' salvo em: {save_path}")

    def _plot_ml_dashboard(self):
        """
        PLANO A (Plotly): Dashboard de Features de ML
        """
        print("Executando plot 'ml_dashboard' (Plotly)...")
        try:
            df = self.repo.get_processed_dataframe("ml_features.csv")
        except FileNotFoundError as e:
            print(f"Erro ao plotar: {e}. Pule esta etapa.")
            return

        fig = go.Figure()
        
        # Adiciona altitude relativa
        fig.add_trace(go.Scatter(x=df['Time_s'], y=df['RelAlt_m_mean'], name='Altitude Relativa (Média)'))
        # Adiciona variância do LIDAR
        fig.add_trace(go.Scatter(x=df['Time_s'], y=df['TerrAlt_m_var'], name='Variância do Terreno (LIDAR)', yaxis='y2'))
        
        fig.update_layout(
            title='Dashboard de Features (Altitude e Variância do Terreno)',
            xaxis_title='Tempo (s)',
            yaxis_title='Altitude Relativa (m)',
            yaxis2=dict(
                title='Variância do LIDAR (m^2)',
                overlaying='y',
                side='right'
            ),
            legend_title='Features'
        )
        
        save_path = os.path.join(self.plot_dir, f'plot_ml_dashboard_{self.repo.log_id}.png')
        fig.write_image(save_path)
        print(f"Plot 'ml_dashboard' salvo em: {save_path}")


    def _plot_outlier_analysis(self):
        """
        PLANO A (Plotly): Análise de Outliers
        """
        print("Executando plot 'outlier_analysis' (Plotly)...")
        try:
            df = self.repo.get_processed_dataframe("ml_features.csv")
        except FileNotFoundError as e:
            print(f"Erro ao plotar: {e}. Pule esta etapa.")
            return

        # Plota a pontuação Z-Score da altitude relativa
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time_s'], y=df['RelAlt_m_zscore'], name='Z-Score (Altitude Relativa)'))
        
        # Adiciona linhas para outliers (ex: Z-Score > 3 ou < -3)
        fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Outlier (+3 std)")
        fig.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="Outlier (-3 std)")
        
        fig.update_layout(
            title='Análise de Outliers (Z-Score)',
            xaxis_title='Tempo (s)',
            yaxis_title='Z-Score'
        )
        
        save_path = os.path.join(self.plot_dir, f'plot_outlier_analysis_{self.repo.log_id}.png')
        fig.write_image(save_path)
        print(f"Plot 'outlier_analysis' salvo em: {save_path}")