"""
Etapa 6 – Caracterización Geométrica y Fractal
-----------------------------------------------
Blueprint DigPatho: Antes de cualquier modificación, se calculan las
propiedades originales de la célula. Este vector descriptor actúa como
restricción matemática en la Etapa 7 para que las células sintéticas
sean biológicamente plausibles.

Métricas calculadas:
  Geométricas: Área, Perímetro, Circularidad, Convexidad, Solidez.
  Fractales:   Dimensión Fractal (Box-Counting), Lacunaridad.
  Textura:     Intensidad media H y DAB (HER2), estadísticos de color.
"""
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(__file__))
from macenko import separate_stains


# =============================================================================
# MÉTRICAS GEOMÉTRICAS
# =============================================================================

def compute_geometric_features(contour, mask):
    """Calcula el vector de características geométricas de la célula."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Circularidad (1.0 = círculo perfecto)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-9)
    
    # Convexidad: ratio perímetro del casco convexo / perímetro real
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    convexity = hull_perimeter / (perimeter + 1e-9)
    
    # Solidez: ratio área real / área del casco convexo
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-9)
    
    # Bounding box aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = min(w, h) / (max(w, h) + 1e-9)
    
    return {
        "area_px":      round(float(area), 4),
        "perimeter_px": round(float(perimeter), 4),
        "circularity":  round(float(circularity), 6),
        "convexity":    round(float(convexity), 6),
        "solidity":     round(float(solidity), 6),
        "aspect_ratio": round(float(aspect_ratio), 6),
        "bbox_w":       int(w),
        "bbox_h":       int(h)
    }


# =============================================================================
# DIMENSIÓN FRACTAL (BOX-COUNTING ALGORITHM)
# =============================================================================

def compute_fractal_dimension(mask_binary):
    """
    Calcula la Dimensión Fractal del contorno celular usando el
    algoritmo de Box-Counting.
    
    D_f = -lim(log N / log ε) donde N es el número de cajas que
    contienen parte del contorno y ε es el tamaño de la caja.
    
    Un contorno circular liso tiene D_f ≈ 1.0.
    Una membrana tumoral agresiva puede llegar a D_f ≈ 1.5-1.7.
    """
    # Trabajar sobre el contorno binario (edges)
    edges = cv2.Canny(mask_binary, 50, 150)
    pixels = np.argwhere(edges > 0)
    
    if len(pixels) == 0:
        return 1.0, {}
    
    # Normalizar al tamaño de la imagen
    H, W = edges.shape
    max_dim = max(H, W)
    
    # Escalas de cajas (progresión log de 2 a 128px)
    box_sizes = [2, 4, 8, 16, 32, 64, min(128, max_dim // 2)]
    box_sizes = [s for s in box_sizes if s < max_dim]
    
    counts = []
    for box_size in box_sizes:
        # Cuántas cajas (grid) contienen al menos un píxel del contorno
        sub = (pixels // box_size)
        count = len(np.unique(sub, axis=0))
        counts.append(count)
        
    counts = np.array(counts, dtype=np.float64)
    sizes  = np.array(box_sizes, dtype=np.float64)
    
    # Regresión lineal en escala log-log para obtener la pendiente = D_f
    log_sizes  = np.log(1.0 / sizes)
    log_counts = np.log(counts + 1e-9)
    
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dim = coeffs[0]
    
    return round(max(1.0, min(2.0, float(fractal_dim))), 6), {
        "box_sizes":  [int(s) for s in box_sizes],
        "box_counts": [int(c) for c in counts]
    }


# =============================================================================
# LACUNARIDAD
# =============================================================================

def compute_lacunarity(mask_binary, box_sizes=None):
    """
    Calcula la Lacunaridad usando el método de Box-Counting deslizante.
    
    La Lacunaridad describe la 'textura' del espacio vacío del fractal.
    Λ ≈ 1.0: fractal homogéneo.
    Λ >> 1.0: fractal muy irregular con grandes espacios vacíos (membrana agresiva).
    
    Λ = (σ_M / μ_M)² + 1
    donde M es la masa (píxeles activos) en cada caja.
    """
    if box_sizes is None:
        box_sizes = [4, 8, 16, 32]
        
    binary = (mask_binary > 0).astype(np.uint8)
    lacunarities = {}
    
    for box_size in box_sizes:
        masses = []
        H, W = binary.shape
        for y in range(0, H - box_size + 1, box_size):
            for x in range(0, W - box_size + 1, box_size):
                box = binary[y:y+box_size, x:x+box_size]
                masses.append(np.sum(box))
                
        masses = np.array(masses, dtype=np.float64)
        if np.mean(masses) < 1e-6:
            lacunarities[f"box_{box_size}"] = 1.0
            continue
            
        lam = (np.std(masses) / (np.mean(masses) + 1e-9)) ** 2 + 1.0
        lacunarities[f"box_{box_size}"] = round(float(lam), 6)
        
    # Lacunaridad global promediada
    mean_lac = float(np.mean(list(lacunarities.values())))
    return round(mean_lac, 6), lacunarities


# =============================================================================
# CONTENIDO DE MINKOWSKI Y FUNCIÓN ZETA TUBULAR (DIMENSIONES COMPLEJAS)
# =============================================================================

def compute_minkowski_and_zeta(mask_binary, r_max=15):
    """
    Calcula el Contenido de Minkowski Superior e Inferior, la Lacunaridad de Minkowski,
    y estima los polos complejos de la función zeta tubular ajustando un modelo oscilatorio
    sobre la frontera de la máscara binaria.
    """
    # 1. Obtener la frontera de la máscara (membrana)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(mask_binary, kernel, iterations=1)
    boundary = cv2.subtract(mask_binary, eroded)
    
    if np.sum(boundary > 0) == 0:
        return {
            "minkowski_dim": 1.0,
            "upper_minkowski": 1.0,
            "lower_minkowski": 1.0,
            "minkowski_lacunarity": 0.0,
            "complex_omega": 0.0,
            "complex_amplitude": 0.0
        }
    
    # 2. Transformada de Distancia Euclidiana (EDT) a la frontera
    dist_map = distance_transform_edt(boundary == 0)
    
    # 3. Medir el volumen tubular V(A_r) para r = 1, 2, ..., r_max
    radii = np.arange(1, r_max + 1, dtype=np.float64)
    volumes = np.zeros_like(radii)
    for i, r in enumerate(radii):
        volumes[i] = np.sum(dist_map <= r)
        
    # 4. Ajustar V(A_r) = c * r^(2-D) en escala log-log para la Dimensión de Minkowski D
    log_r = np.log(radii)
    log_v = np.log(volumes + 1e-9)
    
    coeffs = np.polyfit(log_r, log_v, 1)
    slope = coeffs[0]
    D = 2.0 - slope
    D = max(1.0, min(2.0, float(D)))
    
    # 5. Calcular la serie de contenido local de Minkowski M(r) = V(A_r) / r^(2-D)
    exponent = 2.0 - D
    m_r = volumes / (radii ** exponent + 1e-9)
    
    # 6. Extraer contenido superior e inferior para radios estables r >= 2
    stable_m = m_r[1:] if len(m_r) > 1 else m_r
    upper_M = float(np.max(stable_m))
    lower_M = float(np.min(stable_m))
    mink_lac = upper_M - lower_M
    
    # 7. Ajustar el modelo oscilatorio para estimar la dimensión compleja
    # M(r) = a + b * cos(omega * ln(r) + phase)
    def osc_model(r, a, b, omega, phase):
        return a + b * np.cos(omega * np.log(r) + phase)
    
    a_init = np.mean(stable_m)
    b_init = (upper_M - lower_M) / 2.0
    omega_init = 1.0
    phase_init = 0.0
    
    p0 = [a_init, b_init, omega_init, phase_init]
    
    bounds = (
        [0.0, 0.0, 0.0, -np.pi],
        [np.inf, np.inf, 20.0, np.pi]
    )
    
    omega = 0.0
    amplitude = 0.0
    try:
        fit_r = radii[1:] if len(radii) > 1 else radii
        fit_m = stable_m
        popt, _ = curve_fit(osc_model, fit_r, fit_m, p0=p0, bounds=bounds, maxfev=2000)
        a_fit, b_fit, omega_fit, phase_fit = popt
        omega = float(omega_fit)
        amplitude = float(b_fit)
    except Exception:
        pass
        
    return {
        "minkowski_dim": round(D, 6),
        "upper_minkowski": round(upper_M, 6),
        "lower_minkowski": round(lower_M, 6),
        "minkowski_lacunarity": round(mink_lac, 6),
        "complex_omega": round(omega, 6),
        "complex_amplitude": round(amplitude, 6)
    }


# =============================================================================
# CARACTERÍSTICAS DE TEXTURA Y TINCIÓN (HER2)
# =============================================================================

def compute_staining_features(cell_rgb, mask):
    """
    Extrae características de color basadas en la separación de Macenko.
    Solo se analiza la región dentro de la máscara.
    """
    # Aplicar máscara para ignorar el fondo transparente
    masked_rgb = cell_rgb.copy()
    mask_bool = mask > 0
    
    if not np.any(mask_bool):
        return {}
    
    try:
        concentrations, _ = separate_stains(masked_rgb)
        h_vals  = concentrations[:, :, 0][mask_bool]
        dab_vals = concentrations[:, :, 1][mask_bool]
        
        return {
            "h_mean":    round(float(np.mean(h_vals)),   6),
            "h_std":     round(float(np.std(h_vals)),    6),
            "dab_mean":  round(float(np.mean(dab_vals)), 6),
            "dab_std":   round(float(np.std(dab_vals)),  6),
            "dab_max":   round(float(np.max(dab_vals)),  6),  # Intensidad máxima de HER2
        }
    except Exception:
        return {}


# =============================================================================
# ORQUESTADOR COMPLETO DE CARACTERIZACIÓN
# =============================================================================

def characterize_cell(cell_data):
    """
    Calcula el vector completo de características para una célula.
    Recibe un dict con keys: rgba, mask, contour.
    Retorna el vector de características como dict.
    """
    rgba   = cell_data["rgba"]
    mask   = cell_data["mask"]
    contour = cell_data["contour"]
    
    cell_rgb = rgba[:, :, :3]
    
    # 1. Geométricas
    geo  = compute_geometric_features(contour, mask)
    
    # 2. Dimensión Fractal (Box-Counting)
    df, df_details = compute_fractal_dimension(mask)
    
    # 3. Lacunaridad
    lac, lac_details = compute_lacunarity(mask)
    
    # 3.1 Minkowski y Zetas (Dimensiones Complejas)
    mink_zeta = compute_minkowski_and_zeta(mask)
    
    # 4. Tinción
    stain = compute_staining_features(cell_rgb, mask)
    
    vector = {
        "id":                 cell_data["id"],
        "global_x":           cell_data.get("global_x", 0),
        "global_y":           cell_data.get("global_y", 0),
        "geometric":          geo,
        "fractal_dimension":  df,
        "lacunarity":         lac,
        "minkowski_dim":      mink_zeta["minkowski_dim"],
        "upper_minkowski":    mink_zeta["upper_minkowski"],
        "lower_minkowski":    mink_zeta["lower_minkowski"],
        "minkowski_lacunarity": mink_zeta["minkowski_lacunarity"],
        "complex_omega":      mink_zeta["complex_omega"],
        "complex_amplitude":  mink_zeta["complex_amplitude"],
        "staining":           stain,
        "_details": {
            "box_counting": df_details,
            "lacunarity":   lac_details
        }
    }
    return vector


# =============================================================================
# EJECUCIÓN DIRECTA (TEST)
# =============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image

    cells_dir = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\extracted_cells"
    output_dir = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico"
    os.makedirs(output_dir, exist_ok=True)

    cell_files = sorted([f for f in os.listdir(cells_dir) if f.endswith(".png") and "_mask" not in f])
    
    print(f"\n{'='*55}")
    print(f"  Etapa 6 – Caracterización Geométrica y Fractal")
    print(f"  {len(cell_files)} células encontradas en extracted_cells/")
    print(f"{'='*55}\n")
    
    all_vectors = []
    
    coords_path = os.path.join(cells_dir, "cell_coords.json")
    coords = {}
    if os.path.exists(coords_path):
        with open(coords_path, "r") as f:
            coords = json.load(f)
            
    for cf in cell_files:
        cell_id = cf.replace(".png", "")
        img_path  = os.path.join(cells_dir, cf)
        mask_path = os.path.join(cells_dir, cell_id + "_mask.png")
        
        if not os.path.exists(mask_path):
            continue
        
        rgba = np.array(Image.open(img_path).convert("RGBA"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Reconstruir el contorno desde la máscara
        contours_found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_found:
            continue
        main_contour = max(contours_found, key=cv2.contourArea)
        
        c_info = coords.get(cell_id, {})
        cell_data = {
            "id": cell_id,
            "rgba": rgba,
            "mask": mask,
            "contour": main_contour,
            "global_x": c_info.get("global_x", 0),
            "global_y": c_info.get("global_y", 0)
        }
        
        vector = characterize_cell(cell_data)
        all_vectors.append(vector)
        
        print(f"  {cell_id}: D_f={vector['fractal_dimension']:.4f} | Lac={vector['lacunarity']:.4f} | "
              f"Area={int(vector['geometric']['area_px'])} | Circ={vector['geometric']['circularity']:.3f} | "
              f"DAB_mean={vector['staining'].get('dab_mean', 0):.4f}")
    
    # Guardar los vectores en JSON (metadata del dataset)
    json_path = os.path.join(output_dir, "feature_vectors.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_vectors, f, indent=2)
    print(f"\n  Vectores guardados en: {json_path}")
    
    # --- Panel de Distribución de Dimensión Fractal ---
    if all_vectors:
        dfs   = [v["fractal_dimension"]                    for v in all_vectors]
        lacs  = [v["lacunarity"]                           for v in all_vectors]
        circs = [v["geometric"]["circularity"]             for v in all_vectors]
        dabs  = [v["staining"].get("dab_mean", 0)          for v in all_vectors]
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].hist(dfs, bins=14, color="#4a90d9", edgecolor="white", linewidth=0.7)
        axes[0].set_title("Dimensión Fractal (D_f)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("D_f")
        axes[0].set_ylabel("Frecuencia")
        axes[0].axvline(np.mean(dfs), color='red', linestyle='--', label=f"μ={np.mean(dfs):.3f}")
        axes[0].legend()
        
        axes[1].hist(lacs, bins=14, color="#7ed321", edgecolor="white", linewidth=0.7)
        axes[1].set_title("Lacunaridad (Λ)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Λ")
        axes[1].axvline(np.mean(lacs), color='red', linestyle='--', label=f"μ={np.mean(lacs):.3f}")
        axes[1].legend()
        
        axes[2].hist(circs, bins=14, color="#f5a623", edgecolor="white", linewidth=0.7)
        axes[2].set_title("Circularidad", fontsize=12, fontweight='bold')
        axes[2].set_xlabel("C = 4πA/P²")
        axes[2].axvline(np.mean(circs), color='red', linestyle='--', label=f"μ={np.mean(circs):.3f}")
        axes[2].legend()
        
        axes[3].hist(dabs, bins=14, color="#bd10e0", edgecolor="white", linewidth=0.7)
        axes[3].set_title("Intensidad DAB (HER2)", fontsize=12, fontweight='bold')
        axes[3].set_xlabel("DAB mean [OD]")
        axes[3].axvline(np.mean(dabs), color='red', linestyle='--', label=f"μ={np.mean(dabs):.3f}")
        axes[3].legend()
        
        plt.suptitle(f"Etapa 6 – Distribución de Características ({len(all_vectors)} células | Grado 3+)",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        panel_path = os.path.join(output_dir, "blueprint_etapa6.png")
        plt.savefig(panel_path, dpi=200, facecolor='white')
        print(f"\n  Panel de distribución guardado en: {panel_path}")
        print(f"\n{'='*55}")
        print(f"  Etapa 6 completada exitosamente.")
        print(f"{'='*55}\n")
