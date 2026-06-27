"""
synthetic_pipeline.py — Pipeline Completo de Generacion Sintetica
==================================================================
Etapas A a F del Blueprint Cientifico DigPatho:
  A: Localizacion de celulas (Macenko + Componentes Conectados)
  B: Segmentacion con SAM 2 (mascara exacta de cada membrana)
  C: Caracterizacion Fractal (D_f Box-Counting, Lacunaridad)
  D: Campo Elastico fBm + Weierstrass ponderado por D_f
  E: Deformacion de la imagen COMPLETA con cv2.remap
  F: Augmentacion de color Macenko sobre la imagen deformada
  Salida: N imagenes sinteticas WSI + coordenadas actualizadas + CSV de etiquetas
"""
import os, sys, json, csv, warnings
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from fractal_field       import build_elastic_field, apply_elastic_deformation, transform_coordinates, lacunarity_color_scale
from macenko             import separate_stains, augment_staining
from fractal_characterizer import compute_fractal_dimension, compute_lacunarity

# ─── Constantes configurables ──────────────────────────────────────────────
ALPHA         = 35      # Intensidad de deformacion elastica
SIGMA         = 8       # Suavizado del campo (mayor = mas organico)
HURST         = 0.70    # Exponente de Hurst del fBm
N_VARIANTS    = 3       # Variantes sinteticas por imagen
# ───────────────────────────────────────────────────────────────────────────


# ==========================================================================
# ETAPA A: LOCALIZACION DE CELULAS (Macenko + Componentes Conectados)
# ==========================================================================

def detect_cell_candidates(image_rgb, prob_threshold=0.55):
    """
    Detecta candidatos a celulas combinando:
    - Canal Hematoxilina de Macenko (nucleos)
    - Varianza local (textura)
    - Umbral Otsu adaptativo
    Retorna lista de (cx, cy) coordenadas globales.
    """
    H, W = image_rgb.shape[:2]

    # Macenko → canal H
    try:
        concentrations, _ = separate_stains(image_rgb)
        h_chan = concentrations[:, :, 0]
    except Exception:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        h_chan = (255 - gray).astype(np.float32) / 255.0

    h_norm = cv2.normalize(h_chan, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(h_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morfologia para limpiar ruido
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 80 or area > 5000:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        perim = 2 * (w + h)
        circ = (4 * np.pi * area) / (perim ** 2 + 1e-6)
        if circ < 0.15:
            continue
        cx, cy = centroids[i]
        candidates.append((float(cx), float(cy)))

    print(f"  [Etapa A] {len(candidates)} candidatos detectados.")
    return candidates


# ==========================================================================
# ETAPA B: SEGMENTACION SAM 2
# ==========================================================================

def segment_with_sam2(image_rgb, candidates, model_name="sam2.1_t.pt"):
    """
    Usa SAM 2 para obtener mascaras precisas de cada candidato.
    Retorna lista de (mask_uint8, cx, cy).
    """
    from ultralytics import SAM
    model = SAM(model_name)

    H, W = image_rgb.shape[:2]
    results_list = []
    pad = 50

    for (cx, cy) in candidates:
        # Parche con contexto alrededor del candidato
        x1 = max(0, int(cx) - pad)
        y1 = max(0, int(cy) - pad)
        x2 = min(W, int(cx) + pad)
        y2 = min(H, int(cy) + pad)
        patch = image_rgb[y1:y2, x1:x2]
        ph, pw = patch.shape[:2]

        local_cx = cx - x1
        local_cy = cy - y1

        try:
            res = model.predict(
                source=patch,
                points=[[local_cx, local_cy]],
                labels=[1],
                verbose=False
            )
        except Exception:
            continue

        if not res or res[0].masks is None or len(res[0].masks.data) == 0:
            continue

        mask_local = res[0].masks.data[0].cpu().numpy()
        if mask_local.shape != (ph, pw):
            mask_local = cv2.resize(mask_local, (pw, ph), interpolation=cv2.INTER_NEAREST)

        # Reconstruir mascara en coordenadas globales
        mask_global = np.zeros((H, W), dtype=np.uint8)
        mask_global[y1:y2, x1:x2] = (mask_local * 255).astype(np.uint8)

        # Validacion geometrica minima
        area = np.sum(mask_local > 0.5)
        if area < 150 or area > 10000:
            continue

        results_list.append((mask_global, cx, cy))

    print(f"  [Etapa B] {len(results_list)} mascaras SAM 2 validas.")
    return results_list


# ==========================================================================
# ETAPA C: CARACTERIZACION FRACTAL
# ==========================================================================

def characterize_cells(masks_with_coords):
    """
    Calcula D_f y Lacunaridad para cada celula.
    Retorna lista de (mask, cx, cy, df, lacunarity).
    """
    characterized = []
    for (mask, cx, cy) in masks_with_coords:
        df, _ = compute_fractal_dimension(mask)
        lac, _ = compute_lacunarity(mask)
        characterized.append((mask, cx, cy, df, lac))

    if characterized:
        dfs = [c[3] for c in characterized]
        print(f"  [Etapa C] D_f media={np.mean(dfs):.3f}  min={np.min(dfs):.3f}  max={np.max(dfs):.3f}")
    return characterized


# ==========================================================================
# ETAPA D+E: CAMPO ELASTICO + DEFORMACION COMPLETA
# ==========================================================================

def generate_synthetic_wsi(image_rgb, characterized_cells, variant_idx=0):
    """
    Genera UNA variante sintetica de la WSI completa.
    Combina:
      - Campo elastico fBm + Weierstrass ponderado por D_f  (Etapa D)
      - cv2.remap sobre imagen completa                     (Etapa E)
      - Augmentacion Macenko global                         (Etapa F)
    Retorna: (imagen_sintetica_rgb, nuevas_coords, dx, dy)
    """
    H, W = image_rgb.shape[:2]
    seed  = 1000 + variant_idx * 137

    # Preparar lista (mask, df) para el campo
    masks_df = [(m, df) for (m, cx, cy, df, lac) in characterized_cells]
    coords   = [(cx, cy) for (_, cx, cy, _, _) in characterized_cells]

    # ── Etapa D: Campo elastico ───────────────────────────────────────────
    map_x, map_y, dx, dy = build_elastic_field(
        H, W,
        masks_df_list=masks_df,
        alpha=ALPHA,
        sigma=SIGMA,
        hurst=HURST,
        use_weierstrass=True,
        seed=seed
    )

    # ── Etapa E: Deformacion completa ─────────────────────────────────────
    warped_rgb = apply_elastic_deformation(image_rgb, map_x, map_y)

    # ── Etapa F: Macenko augmentation ─────────────────────────────────────
    # La variacion de color usa la Lacunaridad media como modulador
    mean_lac = np.mean([lac for (_, _, _, _, lac) in characterized_cells]) if characterized_cells else 1.5
    h_s, dab_s = lacunarity_color_scale(mean_lac, base_dab_scale=1.0)

    rng = np.random.default_rng(seed)
    h_scale   = h_s   * rng.uniform(0.85, 1.15)
    dab_scale = dab_s * rng.uniform(0.80, 1.20)
    bright    = rng.uniform(0.92, 1.08)

    synthetic_rgb = augment_staining(warped_rgb,
                                     h_scale=h_scale,
                                     dab_scale=dab_scale,
                                     brightness_scale=bright)

    # Transformar coordenadas con el mismo campo
    new_coords = transform_coordinates(coords, dx, dy, (H, W))

    return synthetic_rgb, new_coords, dx, dy


# ==========================================================================
# EXPORTACION: Imagen + CSV de etiquetas
# ==========================================================================

def save_variant(output_dir, base_name, variant_idx, synthetic_rgb,
                 new_coords, characterized_cells, dx, dy, draw_points=False):
    """
    Guarda la variante sintetica + etiquetas CSV + campo de deformacion.
    """
    os.makedirs(output_dir, exist_ok=True)
    sid = f"{base_name}_syn{variant_idx:02d}"

    # Imagen limpia (para entrenamiento)
    img_path = os.path.join(output_dir, f"{sid}.png")
    Image.fromarray(synthetic_rgb).save(img_path)

    # Imagen con puntos (solo para auditoria del patologo)
    if draw_points:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(synthetic_rgb.shape[1]/100, synthetic_rgb.shape[0]/100), dpi=100)
        ax.imshow(synthetic_rgb)
        for (x, y) in new_coords:
            ax.plot(x, y, 'o', color='#00FF88', markersize=4, alpha=0.8)
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        audit_path = os.path.join(output_dir, f"{sid}_auditoria.png")
        plt.savefig(audit_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # CSV de etiquetas (coordenadas actualizadas)
    csv_path = os.path.join(output_dir, f"{sid}_labels.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "df_original", "lacunarity"])
        for (coord, cell) in zip(new_coords, characterized_cells):
            x_new, y_new = coord
            _, _, _, df, lac = cell
            writer.writerow([round(x_new, 2), round(y_new, 2),
                             round(df, 4), round(lac, 4)])

    # Campo de deformacion (para reproducibilidad)
    field_path = os.path.join(output_dir, f"{sid}_warpfield.npz")
    np.savez_compressed(field_path, dx=dx, dy=dy)

    return sid, img_path


# ==========================================================================
# COMPARATIVA VISUAL
# ==========================================================================

def save_comparison(original_rgb, variants, output_path, base_name):
    """
    Genera collage comparativo: original + N variantes.
    Muestra los puntos de celulas sobre cada imagen.
    """
    n = len(variants) + 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    # Original con puntos
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0].axis("off")

    for i, (synth_rgb, new_coords, info) in enumerate(variants):
        axes[i + 1].imshow(synth_rgb)
        # Pintar los puntos actualizados
        for (x, y) in new_coords:
            axes[i + 1].plot(x, y, 'o', color='#00FF88', markersize=3, alpha=0.7)
        title = f"Sintetica {i+1}\nalpha={info['alpha']}  seed={info['seed']}"
        axes[i + 1].set_title(title, fontsize=10)
        axes[i + 1].axis("off")

    plt.suptitle(f"DigPatho — Pipeline Sintetico: {base_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  Comparativa guardada: {output_path}")


# ==========================================================================
# PIPELINE PRINCIPAL
# ==========================================================================

def run(image_path, output_dir, n_variants=N_VARIANTS, alpha=ALPHA, draw_points=False):
    """
    Ejecuta el pipeline completo A→F sobre una imagen WSI.
    """
    global ALPHA
    ALPHA = alpha

    print(f"\n{'='*60}")
    print(f"  DigPatho — Pipeline Sintetico")
    print(f"  Imagen : {os.path.basename(image_path)}")
    print(f"  Variantes: {n_variants}   Alpha: {alpha}")
    print(f"{'='*60}\n")

    # Cargar imagen
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    H, W = image_rgb.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    synth_dir = os.path.join(output_dir, "dataset_sintetico")

    # A: Detectar candidatos
    candidates = detect_cell_candidates(image_rgb)
    if not candidates:
        print("  [ERROR] No se detectaron candidatos. Revisar imagen.")
        return

    # B: SAM 2
    print("  [Etapa B] Segmentando con SAM 2...")
    masks_with_coords = segment_with_sam2(image_rgb, candidates)
    if not masks_with_coords:
        print("  [AVISO] SAM 2 no produjo mascaras. Usando candidatos directos.")
        masks_with_coords = [
            (np.zeros((H, W), dtype=np.uint8), cx, cy)
            for (cx, cy) in candidates
        ]

    # C: Caracterizacion fractal
    print("  [Etapa C] Calculando Dimension Fractal y Lacunaridad...")
    characterized = characterize_cells(masks_with_coords)

    # D+E+F: Generar N variantes
    variants_info = []
    for vi in range(n_variants):
        print(f"  [Etapas D-E-F] Generando variante {vi+1}/{n_variants}...")
        synth_rgb, new_coords, dx, dy = generate_synthetic_wsi(
            image_rgb, characterized, variant_idx=vi
        )

        sid, img_path = save_variant(
            synth_dir, base_name, vi, synth_rgb, new_coords, characterized, dx, dy, draw_points=draw_points
        )
        variants_info.append((synth_rgb, new_coords, {"alpha": ALPHA, "seed": 1000 + vi * 137}))
        print(f"    Guardado: {sid}.png + _labels.csv + _warpfield.npz")

    # Comparativa visual
    comp_path = os.path.join(output_dir, f"{base_name}_comparativa.png")
    save_comparison(image_rgb, variants_info, comp_path, base_name)

    print(f"\n{'='*60}")
    print(f"  Pipeline completado.")
    print(f"  Dataset: {synth_dir}")
    print(f"  Comparativa: {comp_path}")
    print(f"{'='*60}\n")

    return comp_path


if __name__ == "__main__":
    IMAGE  = r"E:\Genaro\Desktop\Digital pathologies\no entrar\scripts\3+\3+\3Her2.jpg"
    OUTPUT = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico_v2"
    run(IMAGE, OUTPUT, n_variants=3, alpha=35)
