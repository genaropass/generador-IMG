"""
Etapas 7, 8 y 9 – Generación Sintética + Validación + Exportación
------------------------------------------------------------------
Blueprint DigPatho:
  Etapa 7: Mutar borde (fBm) + tinción (Macenko) dentro de los límites del
           vector de características calculado en la Etapa 6.
  Etapa 8: Validar cada sintética con SSIM y error de Dimensión Fractal.
  Etapa 9: Exportar cell_XXXXXX.png + mask + metadata.json por cada muestra.
"""
import os, sys, json, warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from fractal_core       import apply_fractal_mutation
from macenko            import generate_stain_variants
from tps_warping        import warp_image_tps
from fractal_characterizer import compute_fractal_dimension

CELLS_DIR   = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\extracted_cells"
VECTORS_JSON= r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\feature_vectors.json"
OUTPUT_DIR  = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\dataset_sintetico"
N_VARIANTS  = 3       # Variantes sintéticas por célula real
SSIM_MIN    = 0.25    # Umbral mínimo de similitud (bajo = muy distorcionada)
DF_MAX_ERR  = 0.30    # Error máximo permitido en Dimensión Fractal

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar vectores de características (restricciones biológicas)
with open(VECTORS_JSON, encoding="utf-8") as f:
    feature_vectors = {v["id"]: v for v in json.load(f)}


def contour_from_mask(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def stage7_generate(cell_rgb, mask, contour, feat_vector, seed):
    """
    Etapa 7: Genera una variante sintética guiada por el vector de restricciones.
    Retorna imagen RGBA sintética.
    """
    np.random.seed(seed)
    df_orig = feat_vector["fractal_dimension"]

    # Obtener puntos del contorno (y, x)
    pts = contour.reshape(-1, 2)           # (N, x_y)
    src_pts = np.column_stack([pts[:,1], pts[:,0]])   # (N, y, x)

    # Escalar noise_strength en función de D_f original
    # D_f más alto → célula ya agresiva → mutación más suave para no pasarnos
    strength = np.interp(df_orig, [1.0, 1.5], [12.0, 5.0])
    persistence = np.interp(df_orig, [1.0, 1.5], [0.45, 0.65])

    contour_xy = np.column_stack([src_pts[:,1], src_pts[:,0]])  # (N, x, y) para fractal_core
    mutated_xy = apply_fractal_mutation(
        contour_xy,
        noise_strength=strength,
        octaves=5,
        persistence=persistence
    )
    tgt_pts = np.column_stack([mutated_xy[:,1], mutated_xy[:,0]])  # (y, x)

    # Variación de tinción Macenko
    color_variants = generate_stain_variants(cell_rgb, n_variants=1, seed=seed)
    color_img = color_variants[0]

    # Upscale x4 antes del TPS para evitar artefactos blocky en células pequeñas
    SCALE = 4
    h_orig, w_orig = color_img.shape[:2]
    color_up = cv2.resize(color_img, (w_orig * SCALE, h_orig * SCALE), interpolation=cv2.INTER_CUBIC)
    src_pts_up = src_pts * SCALE
    tgt_pts_up = tgt_pts * SCALE

    # TPS warping a alta resolución
    warped_up = warp_image_tps(color_up, src_pts_up, tgt_pts_up, downsample=4)

    # Devolver al tamaño original con bicúbica
    warped_rgb = cv2.resize(warped_up, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    # Crear imagen RGBA sintética
    rgba_out = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2RGBA)
    rgba_out[:, :, 3] = mask   # Alpha = máscara original
    return rgba_out, warped_rgb, tgt_pts


def stage8_validate(orig_rgb, synth_rgb, orig_mask, df_orig):
    """
    Etapa 8: Validación matemática de la célula sintética.
    Retorna True si pasa los umbrales, junto con las métricas.
    """
    # Redimensionar al mismo tamaño si difieren (por TPS)
    h, w = orig_rgb.shape[:2]
    if synth_rgb.shape[:2] != (h, w):
        synth_rgb = cv2.resize(synth_rgb, (w, h))

    # SSIM (solo zona de la máscara)
    orig_gray  = cv2.cvtColor(orig_rgb,  cv2.COLOR_RGB2GRAY)
    synth_gray = cv2.cvtColor(synth_rgb, cv2.COLOR_RGB2GRAY)
    win = min(7, h-1, w-1) | 1   # ventana impar válida
    ssim_score = float(ssim(orig_gray, synth_gray, win_size=win, data_range=255))

    # Error en Dimensión Fractal
    synth_mask = (cv2.cvtColor(synth_rgb, cv2.COLOR_RGB2GRAY) > 10).astype(np.uint8) * 255
    df_synth, _ = compute_fractal_dimension(synth_mask)
    df_error = abs(df_synth - df_orig)

    passed = ssim_score >= SSIM_MIN and df_error <= DF_MAX_ERR

    return passed, {
        "ssim":     round(ssim_score, 4),
        "df_orig":  round(df_orig,    4),
        "df_synth": round(df_synth,   4),
        "df_error": round(df_error,   4)
    }


def stage9_export(cell_id, variant_idx, rgba, orig_mask, feat_vector, metrics, global_x, global_y):
    """
    Etapa 9: Exportar la muestra sintética como entidad independiente.
    """
    sample_id = f"{cell_id}_syn{variant_idx:02d}"
    img_path  = os.path.join(OUTPUT_DIR, f"{sample_id}.png")
    mask_path = os.path.join(OUTPUT_DIR, f"{sample_id}_mask.png")
    meta_path = os.path.join(OUTPUT_DIR, f"{sample_id}_metadata.json")

    Image.fromarray(rgba).save(img_path)
    Image.fromarray(orig_mask).save(mask_path)

    metadata = {
        "id":              sample_id,
        "original_cell":   cell_id,
        "global_x":        global_x,
        "global_y":        global_y,
        "geometric":       feat_vector.get("geometric", {}),
        "fractal_dimension_original": feat_vector.get("fractal_dimension"),
        "lacunarity_original":        feat_vector.get("lacunarity"),
        "staining_original":          feat_vector.get("staining", {}),
        "validation":      metrics,
        "generation_params": {
            "n_variants":  N_VARIANTS,
            "ssim_min":    SSIM_MIN,
            "df_max_error":DF_MAX_ERR
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return sample_id


# ==========================================================================
# ORQUESTADOR PRINCIPAL
# ==========================================================================

def run_stages_7_to_9():
    cell_files = sorted([
        f for f in os.listdir(CELLS_DIR)
        if f.endswith(".png") and "_mask" not in f
    ])

    print(f"\n{'='*55}")
    print(f"  Etapas 7-8-9  |  {len(cell_files)} células base  |  {N_VARIANTS} variantes c/u")
    print(f"{'='*55}\n")

    accepted_total = 0
    rejected_total = 0
    preview_cells  = []   # Para el collage visual final

    for cf in cell_files:
        cell_id   = cf.replace(".png", "")
        img_path  = os.path.join(CELLS_DIR, cf)
        mask_path = os.path.join(CELLS_DIR, cell_id + "_mask.png")

        if not os.path.exists(mask_path):
            continue

        feat = feature_vectors.get(cell_id)
        if feat is None:
            continue

        rgba     = np.array(Image.open(img_path).convert("RGBA"))
        mask     = np.array(Image.open(mask_path).convert("L"))
        cell_rgb = rgba[:, :, :3]
        contour  = contour_from_mask(mask)
        if contour is None or len(contour) < 8:
            continue

        df_orig  = feat["fractal_dimension"]
        gx, gy   = feat.get("global_x", 0), feat.get("global_y", 0)

        for vi in range(N_VARIANTS):
            seed = hash(cell_id + str(vi)) % (2**31)

            # Etapa 7: Generar
            rgba_syn, rgb_syn, _ = stage7_generate(cell_rgb, mask, contour, feat, seed)

            # Etapa 8: Validar
            passed, metrics = stage8_validate(cell_rgb, rgb_syn, mask, df_orig)

            if passed:
                # Etapa 9: Exportar
                sid = stage9_export(cell_id, vi, rgba_syn, mask, feat, metrics, gx, gy)
                accepted_total += 1
                if len(preview_cells) < 12:
                    preview_cells.append((cell_rgb, rgb_syn, metrics))
                print(f"  OK {sid}  SSIM={metrics['ssim']:.3f}  dDf={metrics['df_error']:.3f}")
            else:
                rejected_total += 1

    print(f"\n{'='*55}")
    print(f"  ACEPTADAS: {accepted_total}  |  RECHAZADAS: {rejected_total}")
    print(f"  Dataset en: {OUTPUT_DIR}")
    print(f"{'='*55}\n")

    # Collage visual comparativo (Original vs Sintética)
    if preview_cells:
        n = len(preview_cells)
        fig, axes = plt.subplots(n, 2, figsize=(8, n * 2.5))
        if n == 1:
            axes = [axes]
        for i, (orig, synth, m) in enumerate(preview_cells):
            axes[i][0].imshow(orig);  axes[i][0].axis("off")
            axes[i][0].set_title("Original", fontsize=9)
            axes[i][1].imshow(synth); axes[i][1].axis("off")
            axes[i][1].set_title(f"Sintética  SSIM={m['ssim']:.2f}", fontsize=9)
        plt.suptitle(f"Etapas 7-8-9: {accepted_total} células sintéticas válidas", fontsize=13)
        plt.tight_layout()
        out = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico\blueprint_etapa789.png"
        plt.savefig(out, dpi=150, facecolor="white")
        print(f"  Collage comparativo: {out}")
        plt.close(fig)


if __name__ == "__main__":
    run_stages_7_to_9()
