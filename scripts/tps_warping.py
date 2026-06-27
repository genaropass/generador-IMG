"""
Capa 3: Deformacion Geometrica de Pixeles (Thin Plate Spline - TPS)
--------------------------------------------------------------------
Version vectorizada con NumPy. Sin loops anidados -> mucho mas rapido.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.ndimage import map_coordinates
from scipy.interpolate import RectBivariateSpline


# =============================================================================
# 1. TPS VECTORIZADO
# =============================================================================

def tps_kernel_matrix(src, dst=None):
    """
    Calcula la matriz kernel TPS de forma vectorizada.
    Si dst es None, calcula la matriz cuadrada src x src.
    """
    if dst is None:
        dst = src
    # Diferencias: (N, M, 2)
    diff = src[:, np.newaxis, :] - dst[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)  # (N, M)
    eps = 1e-12
    K = r2 * np.log(r2 + eps)
    K[r2 < eps] = 0.0
    return K


def compute_tps_weights(source_pts, target_pts):
    """
    Calcula los pesos TPS de forma vectorizada (sin loops).
    """
    N = source_pts.shape[0]
    K = tps_kernel_matrix(source_pts)
    P = np.hstack([np.ones((N, 1)), source_pts])

    top    = np.hstack([K, P])
    bottom = np.hstack([P.T, np.zeros((3, 3))])
    L = np.vstack([top, bottom])

    Y = np.vstack([target_pts, np.zeros((3, 2))])

    coeffs = np.linalg.lstsq(L, Y, rcond=None)[0]
    W = coeffs[:N]
    a = coeffs[N:]
    return W, a


def apply_tps_vectorized(query_pts, source_pts, W, a):
    """
    Aplica TPS a todos los puntos de consulta de forma vectorizada.
    """
    K = tps_kernel_matrix(query_pts, source_pts)  # (Q, N)
    P = np.hstack([np.ones((query_pts.shape[0], 1)), query_pts])
    return K @ W + P @ a


def warp_image_tps(image_rgb, source_pts, target_pts, downsample=8):
    """
    Aplica la deformacion TPS a toda la imagen de forma vectorizada.
    """
    H, W = image_rgb.shape[:2]
    H_d = max(H // downsample, 10)
    W_d = max(W // downsample, 10)
    scale = 1.0 / downsample

    source_pts_d = source_pts * scale
    target_pts_d = target_pts * scale
    displacements_d = target_pts_d - source_pts_d

    print("Calculando pesos TPS (vectorizado)...")
    W_coef, a_coef = compute_tps_weights(source_pts_d, displacements_d)

    # Grilla reducida
    ys = np.arange(H_d, dtype=np.float32)
    xs = np.arange(W_d, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
    query_pts = np.column_stack([grid_y.ravel(), grid_x.ravel()])

    print(f"Aplicando TPS sobre grilla {H_d}x{W_d} ({len(query_pts)} puntos)...")
    displacements = apply_tps_vectorized(query_pts, source_pts_d, W_coef, a_coef)

    map_y_d = (query_pts[:, 0] - displacements[:, 0]).reshape(H_d, W_d)
    map_x_d = (query_pts[:, 1] - displacements[:, 1]).reshape(H_d, W_d)

    # Interpolar al tamaño original
    y_full = np.linspace(0, H_d - 1, H)
    x_full = np.linspace(0, W_d - 1, W)

    map_y_full = RectBivariateSpline(ys, xs, map_y_d)(y_full, x_full) * downsample
    map_x_full = RectBivariateSpline(ys, xs, map_x_d)(y_full, x_full) * downsample

    map_y_full = np.clip(map_y_full, 0, H - 1)
    map_x_full = np.clip(map_x_full, 0, W - 1)

    # Remap por canal
    result = np.zeros_like(image_rgb)
    for c in range(3):
        result[:, :, c] = map_coordinates(
            image_rgb[:, :, c].astype(np.float64),
            [map_y_full, map_x_full],
            order=1,
            mode='reflect'
        )

    return np.clip(result, 0, 255).astype(np.uint8)


# =============================================================================
# 2. PRUEBA CON CELULA MOCK
# =============================================================================

def run_mock_test(output_dir):
    from fractal_core import generate_mock_cell, apply_fractal_mutation

    os.makedirs(output_dir, exist_ok=True)
    SIZE = 200

    # Imagen mock con nucleo y citoplasma
    image_mock = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 220
    cy, cx = SIZE // 2, SIZE // 2
    Y, X = np.ogrid[:SIZE, :SIZE]
    image_mock[(X - cx)**2 + (Y - cy)**2 <= 60**2] = [210, 170, 180]
    image_mock[(X - cx)**2 + (Y - cy)**2 <= 20**2] = [120, 100, 180]

    # Puntos de control: circulo original (N=40, reducido para velocidad)
    N = 40
    theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
    r = 60
    source_pts = np.column_stack([cy + r * np.sin(theta), cx + r * np.cos(theta)])

    # Contorno fractal como destino
    contour_xy = np.column_stack([source_pts[:, 1], source_pts[:, 0]])
    mutated_xy = apply_fractal_mutation(contour_xy, noise_strength=8.0, octaves=5, persistence=0.6)
    target_pts  = np.column_stack([mutated_xy[:, 1], mutated_xy[:, 0]])

    warped = warp_image_tps(image_mock, source_pts, target_pts, downsample=8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_mock)
    axes[0].plot(source_pts[:, 1], source_pts[:, 0], 'g-', lw=2)
    axes[0].set_title("Célula Original (Mock)")
    axes[0].axis("off")

    axes[1].imshow(warped)
    axes[1].plot(target_pts[:, 1], target_pts[:, 0], 'r-', lw=2)
    axes[1].set_title("Deformación TPS + Fractal")
    axes[1].axis("off")

    plt.tight_layout()
    out = os.path.join(output_dir, "prueba_tps_warping.png")
    plt.savefig(out, dpi=200, facecolor='white')
    print(f"Guardado en: {out}")
    plt.show()


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output_sintetico")
    run_mock_test(output_dir)
