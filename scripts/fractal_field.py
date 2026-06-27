"""
fractal_field.py — Motor Fractal Completo
==========================================
Herramientas fractales usadas en el pipeline:
  1. fBm 2D (Movimiento Browniano Fraccionario) via sintesis espectral
  2. Weierstrass-Mandelbrot noise (segunda fuente fractal, multi-escala)
  3. Mapa de intensidad ponderado por D_f (Dimension Fractal por celula)
  4. Lacunaridad como modulador de la variacion de color Macenko
  5. Campo elastico final: cv2.remap sobre la imagen completa
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2


# =============================================================================
# 1. fBm 2D — SINTESIS ESPECTRAL
# =============================================================================

def fbm_field_2d(H, W, hurst=0.7, seed=None):
    """
    Genera un campo 2D de Movimiento Browniano Fraccionario (fBm) mediante
    sintesis espectral (metodo de Fourier).

    hurst > 0.5: campo persistente (deformaciones suaves, organicas)
    hurst = 0.5: ruido browniano puro
    hurst < 0.5: campo anti-persistente (bordes muy irregulares)

    Para membranas celulares tumorales H=0.65-0.75 es biologicamente realista.
    """
    rng = np.random.default_rng(seed)

    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    fx, fy = np.meshgrid(fx, fy)

    freq = np.sqrt(fx ** 2 + fy ** 2)
    freq[0, 0] = 1.0

    # Espectro de potencia: S(f) propto f^(-2H-2) para fBm 2D isotrópico
    power = freq ** (-(hurst + 1.0))
    power[0, 0] = 0.0

    # Ruido blanco complejo en dominio de frecuencia
    noise = (rng.standard_normal((H, W)) + 1j * rng.standard_normal((H, W)))

    # Campo fBm en dominio espacial
    field = np.real(np.fft.ifft2(noise * np.sqrt(power)))

    # Normalizar a [-1, 1]
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= (std * 3)
    return np.clip(field, -1, 1).astype(np.float32)


# =============================================================================
# 2. WEIERSTRASS-MANDELBROT NOISE — segunda fuente fractal
# =============================================================================

def weierstrass_noise_2d(H, W, D=1.5, N_terms=8, seed=None):
    """
    Genera ruido fractal basado en la funcion de Weierstrass-Mandelbrot.
    W(x,y) = sum_{n=0}^{N} gamma^{(D-2)*n} * sin(gamma^n * (x*cos(phi_n) + y*sin(phi_n)) + psi_n)

    D: Dimension fractal objetivo del campo [1.0, 2.0]
    gamma: base de la progresion de frecuencias (tipicamente 1.5)
    """
    rng = np.random.default_rng(seed)
    gamma = 1.5
    y_coords, x_coords = np.mgrid[0:H, 0:W]
    x_n = x_coords / W
    y_n = y_coords / H

    field = np.zeros((H, W), dtype=np.float64)
    for n in range(N_terms):
        phi   = rng.uniform(0, 2 * np.pi)
        psi   = rng.uniform(0, 2 * np.pi)
        freq  = gamma ** n
        amp   = gamma ** ((D - 2) * n)
        field += amp * np.sin(freq * (x_n * np.cos(phi) + y_n * np.sin(phi)) + psi)

    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= (std * 3)
    return np.clip(field, -1, 1).astype(np.float32)


# =============================================================================
# 3. MAPA DE INTENSIDAD PONDERADO POR D_f
# =============================================================================

def build_df_intensity_map(H, W, masks_df_list, sigma_dilate=15, sigma_blur=8):
    """
    Crea un mapa 2D de intensidad de deformacion guiado por la Dimension Fractal.

    Zonas de membrana con alta D_f (celula agresiva) reciben mayor deformacion.
    Zonas de estroma/fondo reciben deformacion suave.

    masks_df_list: lista de (mask_uint8, df_value)
    Retorna: intensity_map en [0.2, 1.0]
    """
    intensity_map = np.full((H, W), 0.2, dtype=np.float32)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (sigma_dilate, sigma_dilate)
    )

    for mask, df_val in masks_df_list:
        # Normalizar D_f: 1.0 → 0.0, 2.0 → 1.0
        df_norm = float(np.clip((df_val - 1.0) / 1.0, 0, 1))

        dilated = cv2.dilate(mask.astype(np.uint8), kernel)
        cell_contribution = 0.2 + df_norm * 0.8  # [0.2, 1.0]

        intensity_map = np.maximum(
            intensity_map,
            (dilated > 0).astype(np.float32) * cell_contribution
        )

    # Suavizar para evitar discontinuidades
    intensity_map = gaussian_filter(intensity_map, sigma=sigma_blur).astype(np.float32)
    return np.clip(intensity_map, 0.2, 1.0)


# =============================================================================
# 4. MODULADOR DE COLOR POR LACUNARIDAD
# =============================================================================

def lacunarity_color_scale(lacunarity, base_dab_scale=1.0):
    """
    La Lacunaridad describe la heterogeneidad interna de la celula.
    Alta Lacunaridad → mayor variacion de color DAB (mas agresividad).

    Retorna factores de escala para H y DAB en Macenko.
    """
    # Lacunaridad > 2.0 se considera alta
    lac_norm = np.clip((lacunarity - 1.0) / 3.0, 0, 1)

    dab_scale = base_dab_scale * (1.0 + lac_norm * 0.4)   # hasta +40%
    h_scale   = 1.0 - lac_norm * 0.15                      # hasta -15%

    return float(h_scale), float(dab_scale)


# =============================================================================
# 5. CAMPO ELASTICO COMPLETO Y cv2.remap
# =============================================================================

def build_elastic_field(H, W, masks_df_list=None,
                         alpha=35, sigma=8, hurst=0.7,
                         use_weierstrass=True, seed=None):
    """
    Construye el campo de deformacion elastica combinando:
      - fBm 2D espectral (componente principal)
      - Weierstrass-Mandelbrot noise (componente de alta frecuencia)
      - Mapa de intensidad ponderado por D_f

    Retorna: map_x, map_y, dx, dy
    """
    seed_x = seed if seed is not None else 42
    seed_y = seed_x + 1

    # Campo base fBm
    dx_fbm = fbm_field_2d(H, W, hurst=hurst, seed=seed_x)
    dy_fbm = fbm_field_2d(H, W, hurst=hurst, seed=seed_y)

    # Campo Weierstrass como perturbacion de alta frecuencia (30% del total)
    if use_weierstrass and masks_df_list:
        mean_df = np.mean([df for _, df in masks_df_list]) if masks_df_list else 1.3
        dx_w = weierstrass_noise_2d(H, W, D=mean_df, seed=seed_x + 10)
        dy_w = weierstrass_noise_2d(H, W, D=mean_df, seed=seed_y + 10)
        dx = dx_fbm * 0.70 + dx_w * 0.30
        dy = dy_fbm * 0.70 + dy_w * 0.30
    else:
        dx, dy = dx_fbm, dy_fbm

    # Mapa de intensidad por D_f
    if masks_df_list:
        intensity = build_df_intensity_map(H, W, masks_df_list, sigma_blur=sigma)
    else:
        intensity = np.full((H, W), 0.5, dtype=np.float32)

    # Aplicar intensidad y alpha global
    dx = dx * intensity * alpha
    dy = dy * intensity * alpha

    # Suavizado final
    dx = gaussian_filter(dx, sigma=sigma / 2).astype(np.float32)
    dy = gaussian_filter(dy, sigma=sigma / 2).astype(np.float32)

    # Grilla de mapeo
    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = x_grid + dx
    map_y = y_grid + dy

    return map_x, map_y, dx, dy


def apply_elastic_deformation(image_rgb, map_x, map_y):
    """Aplica el campo elastico a la imagen completa via cv2.remap."""
    return cv2.remap(
        image_rgb, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101
    )


def transform_coordinates(coords_xy, dx, dy, image_shape):
    """
    Transforma coordenadas (x, y) de celdas usando el mismo campo de desplazamiento.
    coords_xy: lista de (x, y)
    Retorna: lista de (x', y')
    """
    H, W = dx.shape
    new_coords = []
    for (x, y) in coords_xy:
        xi = int(np.clip(round(x), 0, W - 1))
        yi = int(np.clip(round(y), 0, H - 1))
        x_new = float(np.clip(x + dx[yi, xi], 0, W - 1))
        y_new = float(np.clip(y + dy[yi, xi], 0, H - 1))
        new_coords.append((x_new, y_new))
    return new_coords
