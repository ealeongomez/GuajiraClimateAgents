#!/usr/bin/env python3
# ======================================================
# Project : GuajiraClimateAgents
# Author  : Eder Arley León Gómez
# GitHub  : https://github.com/ealeongomez
# License : MIT
# ======================================================
"""
Generador de predicciones de viento usando modelos LSTM entrenados.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List


# ============================================================================
# CLASES DEL MODELO (copiadas del notebook de entrenamiento)
# ============================================================================

class DenseRFF_PT(nn.Module):
    def __init__(self, Nf, scale=None, gamma=None, normalization=True,
                 function="cos", trainable_scale=True, trainable_W=True,
                 seed=None, kernel='gaussian'):
        super().__init__()
        self.Nf = Nf
        self.gamma = gamma
        self.scale = scale
        self.normalization = normalization
        self.function = function
        self.trainable_scale = trainable_scale
        self.trainable_W = trainable_W
        self.seed = seed
        self.kernel_type = kernel
        self.W = None
        self.b = None
        self.rho_scale = None
        self._eps = 1e-8
        self.bandwidth_history = []

    def _get_random_features_initializer(self, shape, sigma=1.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if self.kernel_type == 'gaussian':
            return np.random.randn(*shape) / sigma
        elif self.kernel_type == 'laplacian':
            return np.random.laplace(loc=0.0, scale=1.0, size=shape) / sigma
        else:
            raise ValueError(f'Unsupported initializer {self.kernel_type}')

    def _ensure_params_initialized(self, device, D):
        if self.W is None:
            if self.gamma is not None:
                sigma = np.sqrt(1.0 / (2 * self.gamma))
            else:
                sigma = 1.0
            if self.scale is None:
                self.scale = sigma
            W_init = self._get_random_features_initializer((D, self.Nf),
                                                           sigma=self.scale,
                                                           seed=self.seed)
            self.W = nn.Parameter(torch.tensor(W_init, dtype=torch.float32, device=device),
                                  requires_grad=self.trainable_W)
            b_init = np.random.uniform(0.0, 2 * np.pi, size=(self.Nf,))
            self.b = nn.Parameter(torch.tensor(b_init, dtype=torch.float32, device=device),
                                  requires_grad=self.trainable_W)
            init_kernel_scale = 1.0
            rho0 = np.log(np.exp(init_kernel_scale) - 1.0)
            self.rho_scale = nn.Parameter(
                torch.tensor([rho0], dtype=torch.float32, device=device),
                requires_grad=self.trainable_scale
            )

    def _kernel_scale(self):
        return F.softplus(self.rho_scale) + self._eps

    def bandwidth_lengthscale(self):
        if self.rho_scale is None or self.scale is None:
            return None
        ks = float(self._kernel_scale().detach().cpu().item())
        return float(self.scale) / ks

    @torch.no_grad()
    def log_bandwidth(self, step):
        ell = self.bandwidth_lengthscale()
        if ell is not None:
            self.bandwidth_history.append((step, float(ell)))

    def forward(self, inputs):
        device = inputs.device
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        elif inputs.dim() != 3:
            raise ValueError(f"Expected [B,T,D], got {inputs.shape}")
        B, T, D = inputs.shape
        self._ensure_params_initialized(device, D)
        kernel_scale = self._kernel_scale()
        proj = torch.matmul(inputs, self.W * kernel_scale) + self.b
        outputs = torch.cos(proj) * np.sqrt(2.0 / self.Nf)
        if self.normalization:
            norm = np.sqrt(self.Nf)
            outputs = outputs / norm
        return outputs.permute(0, 2, 1)


class SpectralDropout1d(nn.Module):
    def __init__(self, p: float = 0.1, channels_last: bool = True):
        super().__init__()
        self.p = p
        self.channels_last = channels_last
        
    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        if self.channels_last:
            B, T, F = x.shape
            mask = (torch.rand(B, 1, F, device=x.device) > self.p).float() / (1.0 - self.p)
            return x * mask
        else:
            B, F, T = x.shape
            mask = (torch.rand(B, F, 1, device=x.device) > self.p).float() / (1.0 - self.p)
            return x * mask


class TemporalSpectralBlock(nn.Module):
    def __init__(self, features, kernel_size=3, dilation=2, p_drop=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(features, features, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(features, features, kernel_size, padding=padding, dilation=1)
        self.norm = nn.LayerNorm(features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        res = x
        y = x.transpose(1, 2)
        y = self.act(self.conv1(y))
        y = self.conv2(y).transpose(1, 2)
        if y.size(1) != res.size(1):
            min_len = min(y.size(1), res.size(1))
            y = y[:, :min_len, :]
            res = res[:, :min_len, :]
        y = self.norm(y + res)
        y = self.act(y)
        return self.drop(y)


class MultiBandRFFEncoder(nn.Module):
    def __init__(self, in_dim=1, bands=(4., 24., 168.), nf_per_band=32,
                 kernel="gaussian", spectral_dropout_p=0.1):
        super().__init__()
        self.bands = nn.ModuleList([
            DenseRFF_PT(Nf=nf_per_band, function="cos",
                        trainable_W=True, trainable_scale=True,
                        kernel=kernel, scale=band)
            for band in bands
        ])
        self.out_dim = nf_per_band * len(self.bands)
        self.norm = nn.LayerNorm(self.out_dim)
        self.spec_do = SpectralDropout1d(p=spectral_dropout_p, channels_last=True)
        self.drop = nn.Dropout(spectral_dropout_p)

    def forward(self, x):
        outs = []
        for b in self.bands:
            z = b(x)
            z = z.transpose(1, 2)
            outs.append(z)
        zcat = torch.cat(outs, dim=2)
        zcat = self.norm(zcat)
        zcat = self.drop(self.spec_do(zcat))
        return zcat


class TemporalHead(nn.Module):
    def __init__(self, hidden, horizon, pool="last"):
        super().__init__()
        self.pool = pool
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, horizon)
        self.act = nn.ReLU()

    def forward(self, H):
        if self.pool == "mean":
            v = H.mean(dim=1)
        else:
            v = H[:, -1, :]
        return self.fc2(self.act(self.fc1(v)))


class RFF_AnyRNN_Forecaster(nn.Module):
    def __init__(self, horizon=24, rnn_type="LSTM",
                 bands=(4., 24., 168.), nf_per_band=64,
                 hidden=96, num_layers=1, bidirectional=False,
                 use_tsb=True, kernel="gaussian",
                 spectral_dropout_p=0.1, pool="last"):
        super().__init__()
        self.enc = MultiBandRFFEncoder(1, bands, nf_per_band, kernel, spectral_dropout_p)
        self.tsb = TemporalSpectralBlock(self.enc.out_dim) if use_tsb else nn.Identity()
        self.rnn = getattr(nn, rnn_type)(
            input_size=self.enc.out_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_hidden = hidden * (2 if bidirectional else 1)
        self.head = TemporalHead(out_hidden, horizon, pool)

    def forward(self, x):
        z = self.enc(x)
        z = self.tsb(z)
        H, _ = self.rnn(z)
        return self.head(H)


# ============================================================================
# GENERADOR DE PREDICCIONES
# ============================================================================

class ForecastGenerator:
    """Genera predicciones de viento para todos los municipios."""
    
    MUNICIPIOS = [
        'albania', 'barrancas', 'distraccion', 'el_molino', 'fonseca',
        'hatonuevo', 'la_jagua_del_pilar', 'maicao', 'manaure', 'mingueo',
        'riohacha', 'san_juan_del_cesar', 'uribia'
    ]
    
    MODEL_NAME_MAP = {
        'albania': 'Albania',
        'barrancas': 'Barrancas',
        'distraccion': 'Distraccion',
        'el_molino': 'El_Molino',
        'fonseca': 'Fonseca',
        'hatonuevo': 'Hatonuevo',
        'la_jagua_del_pilar': 'La_Jagua_del_Pilar',
        'maicao': 'Maicao',
        'manaure': 'Manaure',
        'mingueo': 'Mingueo',
        'riohacha': 'Riohacha',
        'san_juan_del_cesar': 'San_Juan_del_Cesar',
        'uribia': 'Uribia'
    }
    
    def __init__(self, models_dir: str = "data/models/LSTM"):
        """
        Args:
            models_dir: Directorio donde están los modelos .pt
        """
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.normalization_params = {}
        
    def load_models(self):
        """Carga todos los modelos entrenados."""
        print("📥 Cargando modelos de predicción...")
        
        for mun in self.MUNICIPIOS:
            model_name = self.MODEL_NAME_MAP[mun]
            model_file = self.models_dir / f"model_LSTM_{model_name}.pt"
            
            if not model_file.exists():
                print(f"  ⚠️  Modelo no encontrado: {mun} ({model_file})")
                continue
                
            try:
                checkpoint = torch.load(model_file, map_location=self.device)
                
                # Extraer hiperparámetros
                best_params = checkpoint['best_params']
                
                # Instanciar modelo con valores por defecto
                model = RFF_AnyRNN_Forecaster(
                    horizon=24,
                    rnn_type="LSTM",
                    bands=best_params.get('bands', (4., 24., 168.)),
                    nf_per_band=best_params.get('nf_per_band', 64),
                    hidden=best_params.get('hidden', 96),
                    num_layers=best_params.get('num_layers', 1),
                    bidirectional=best_params.get('bidirectional', False),
                    use_tsb=best_params.get('use_tsb', True),
                    kernel=best_params.get('kernel', 'gaussian'),
                    spectral_dropout_p=best_params.get('spectral_dropout_p', 0.1),
                    pool=best_params.get('pool', 'last')
                ).to(self.device)
                
                # Forward dummy para inicializar
                with torch.no_grad():
                    dummy_input = torch.randn(1, 48, 1).to(self.device)
                    _ = model(dummy_input)
                
                # Cargar pesos
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.models[mun] = model
                print(f"  ✅ {mun}")
                
            except Exception as e:
                print(f"  ❌ {mun}: Error - {str(e)}")
                
        print(f"✅ {len(self.models)} modelos cargados\n")
        
    def load_normalization_params(self, conn):
        """Carga parámetros de normalización desde la BD."""
        print("📊 Obteniendo parámetros de normalización...")
        
        cursor = conn.cursor()
        for mun in self.MUNICIPIOS:
            query = f"""
            SELECT MIN(wind_speed_10m) as min_val, MAX(wind_speed_10m) as max_val
            FROM climate_observations
            WHERE municipio = '{mun}' AND wind_speed_10m IS NOT NULL
            """
            cursor.execute(query)
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                self.normalization_params[mun] = {
                    'min': result[0],
                    'max': result[1]
                }
                
        cursor.close()
        print(f"✅ Parámetros cargados para {len(self.normalization_params)} municipios\n")
        
    def get_last_48h_data(self, conn, municipio: str) -> pd.DataFrame:
        """Obtiene últimas 48 horas de datos para un municipio."""
        query = f"""
        SELECT TOP 48 datetime, wind_speed_10m
        FROM climate_observations
        WHERE municipio = '{municipio}'
        ORDER BY datetime DESC
        """
        df = pd.read_sql(query, conn)
        return df.sort_values('datetime').reset_index(drop=True)
        
    def generate_forecast(
        self, 
        wind_data: np.ndarray, 
        municipio: str
    ) -> Tuple[List[float], List[float]]:
        """
        Genera predicción para un municipio.
        
        Args:
            wind_data: Array de 48 valores de velocidad del viento
            municipio: Nombre del municipio
            
        Returns:
            (input_values, predicted_values): Tupla con 48 valores de entrada y 24 predicciones
        """
        if municipio not in self.models:
            raise ValueError(f"Modelo no disponible para {municipio}")
            
        # Normalizar datos
        min_val = self.normalization_params[municipio]['min']
        max_val = self.normalization_params[municipio]['max']
        wind_normalized = (wind_data - min_val) / (max_val - min_val)
        
        # Preparar input
        X_input = torch.tensor(
            wind_normalized, 
            dtype=torch.float32
        ).reshape(1, 48, 1).to(self.device)
        
        # Generar predicción
        with torch.no_grad():
            y_pred_normalized = self.models[municipio](X_input)
        
        # Desnormalizar
        y_pred = y_pred_normalized.detach().cpu().tolist()[0]
        y_pred_denormalized = [
            y * (max_val - min_val) + min_val 
            for y in y_pred
        ]
        
        # Array completo de entrada (48 valores)
        input_values = wind_data.tolist()
        
        return input_values, y_pred_denormalized
        
    def generate_all_forecasts(self, conn) -> Dict[str, pd.DataFrame]:
        """
        Genera predicciones para todos los municipios.
        
        Returns:
            Dict con DataFrames de predicciones por municipio
        """
        print("🔮 Generando predicciones...")
        forecasts = {}
        
        for mun in self.MUNICIPIOS:
            if mun not in self.models:
                print(f"  ⏭️  Saltando {mun} (sin modelo)")
                continue
                
            # Obtener datos históricos
            data_48h = self.get_last_48h_data(conn, mun)
            
            if len(data_48h) < 48:
                print(f"  ⚠️  {mun}: Datos insuficientes ({len(data_48h)}/48)")
                continue
                
            wind_data = data_48h['wind_speed_10m'].values
            last_datetime = data_48h['datetime'].iloc[-1]
            
            # Generar predicción
            input_values, predicted_values = self.generate_forecast(wind_data, mun)
            
            # Crear fecha de inicio (primera hora predicha)
            datetime_inicio = last_datetime + pd.Timedelta(hours=1)
            
            # Crear DataFrame con 1 fila por municipio
            forecasts[mun] = pd.DataFrame({
                'municipio': [mun],
                'datetime_inicio': [datetime_inicio],
                'wind_speed_input': [input_values],      # Array de 48 valores
                'wind_speed_output': [predicted_values]  # Array de 24 valores
            })
            
            print(f"  ✅ {mun}: {len(predicted_values)} horas predichas")
            
        print(f"\n✅ Predicciones generadas para {len(forecasts)} municipios\n")
        return forecasts

