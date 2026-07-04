"""
Capa 2: Separacion y Variacion de Tincion (Algoritmo de Macenko)
-----------------------------------------------------------------
El algoritmo de Macenko descompone matematicamente una imagen HER2 en
sus dos canales de color fundamentales:
  - Hematoxilina (H): tine los nucleos celulares de azul/violeta.
  - DAB (D): tine la membrana HER2 positiva de marron.

Una vez separados, podemos alterar la concentracion de cada canal
de forma independiente para simular diferentes laboratorios, escáneres
y concentraciones de reactivos. Esto es la base de la síntesis de datos.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


# =============================================================================
# 1. MATRICES DE REFERENCIA MACENKO (HE + DAB)
# =============================================================================
# Estos valores son los vectores de absorbancia óptica estándar de la literatura
# para Hematoxilina y DAB en el espacio de densidad óptica (OD).
# Referencia: Macenko et al., ISBI 2009.

# H&E estándar
HE_REFERENCE = np.array([
    [0.5626, 0.7201, 0.4062],
    [0.2159, 0.8012, 0.5581]
])

# IHC / Inmunomarcación (Hematoxilina + DAB)
IHC_REFERENCE = np.array([
    [0.6500, 0.7040, 0.2860],   # Hematoxilina azul/violeta
    [0.2680, 0.5700, 0.7760]    # DAB marrón dorado
])


def calibrate_reference(image_rgb, percentile=99):
    """
    Calibra automáticamente la matriz de referencia Macenko
    a partir de los píxeles más teñidos de la imagen real.
    Usa PCA sobre los píxeles OD con alta absorbancia.
    """
    od = rgb_to_od(image_rgb)
    od_flat = od.reshape(-1, 3)
    # Solo usar píxeles con absorbancia suficiente (no fondo blanco)
    mask = np.linalg.norm(od_flat, axis=1) > 0.15
    od_tissue = od_flat[mask]
    if len(od_tissue) < 100:
        return IHC_REFERENCE  # fallback
    # PCA para encontrar los dos vectores principales de color
    od_mean = od_tissue.mean(axis=0)
    _, _, Vt = np.linalg.svd(od_tissue - od_mean, full_matrices=False)
    # Los dos primeros vectores singulares son los ejes de tinción
    ref = Vt[:2]
    # Normalizar para que apunten hacia densidades positivas
    for i in range(2):
        if ref[i].sum() < 0:
            ref[i] *= -1
    return ref


# =============================================================================
# 2. CONVERSION RGB <-> DENSIDAD OPTICA (OD)
# =============================================================================

def rgb_to_od(image_rgb):
    """
    Convierte una imagen RGB (uint8) a espacio de Densidad Optica (OD).
    OD = -log(I / I0), donde I0 = 255 (luz blanca de referencia).
    Se clampea a 1e-6 para evitar log(0).
    """
    image_rgb = image_rgb.astype(np.float64)
    image_rgb = np.clip(image_rgb, 1e-6, 255.0)
    od = -np.log(image_rgb / 255.0)
    return od


def od_to_rgb(od):
    """
    Convierte densidad optica de vuelta a RGB (uint8).
    """
    rgb = np.exp(-od) * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


# =============================================================================
# 3. SEPARACION DE CANALES (MACENKO SIMPLIFICADO)
# =============================================================================

def separate_stains(image_rgb, reference=HE_REFERENCE):
    """
    Separa la imagen en sus canales de tincion (H y DAB).
    
    Retorna:
        concentrations: array (H, W, 2) con la concentracion de cada tincion
                        en cada pixel. Canal 0 = H, Canal 1 = DAB.
        od: imagen en espacio OD.
    """
    h, w, _ = image_rgb.shape
    od = rgb_to_od(image_rgb)

    # Aplanar a (N, 3) para la multiplicacion matricial
    od_flat = od.reshape(-1, 3)

    # Separar usando pseudoinversa de la matriz de referencia
    # OD = C @ M  =>  C = OD @ pinv(M)
    # M shape: (2, 3), pinv(M) shape: (3, 2)
    # od_flat shape: (N, 3), resultado: (N, 2)
    M = reference                          # (2, 3)
    M_pinv = np.linalg.pinv(M)            # (3, 2)
    concentrations = od_flat @ M_pinv     # (N, 2)
    concentrations = concentrations.reshape(h, w, 2)

    return concentrations, od


def reconstruct_from_stains(concentrations, reference=HE_REFERENCE):
    """
    Reconstruye la imagen RGB a partir de las concentraciones y la referencia.
    """
    h, w, _ = concentrations.shape
    conc_flat = concentrations.reshape(-1, 2)

    # OD reconstruido = Concentracion * Matriz de referencia
    od_reconstructed = conc_flat @ reference
    od_reconstructed = od_reconstructed.reshape(h, w, 3)

    return od_to_rgb(od_reconstructed)


# =============================================================================
# 4. AUMENTACION DE COLOR (EL CORAZON DE LA SINTESIS)
# =============================================================================

def augment_staining(
    image_rgb,
    h_scale=1.0,
    dab_scale=1.0,
    h_shift=0.0,
    dab_shift=0.0,
    brightness_scale=1.0,
    stain_type="ihc"
):
    """
    Aplica variaciones controladas de tincion sobre una imagen HER2.
    
    Parametros:
        h_scale (float):        Factor de escala de Hematoxilina (0.7 = más claro, 1.3 = más oscuro)
        dab_scale (float):      Factor de escala de DAB (0.7 = HER2 debil, 1.3 = HER2 intenso)
        h_shift (float):        Desplazamiento aditivo en H (puede ser negativo)
        dab_shift (float):      Desplazamiento aditivo en DAB
        brightness_scale (float): Simula el brillo general del escaner (0.8 a 1.2)
    
    Retorna:
        Imagen sintética RGB augmentada.
    """
    ref = IHC_REFERENCE if stain_type == "ihc" else HE_REFERENCE
    concentrations, _ = separate_stains(image_rgb, reference=ref)

    # Alterar concentraciones de cada canal independientemente
    synthetic_conc = concentrations.copy()
    synthetic_conc[:, :, 0] = synthetic_conc[:, :, 0] * h_scale + h_shift    # Hematoxilina
    synthetic_conc[:, :, 1] = synthetic_conc[:, :, 1] * dab_scale + dab_shift  # DAB

    # Evitar concentraciones negativas (no fisicamente posibles)
    synthetic_conc = np.clip(synthetic_conc, 0, None)

    # Reconstruir imagen
    synthetic_rgb = reconstruct_from_stains(synthetic_conc, reference=ref)

    # Aplicar factor de brillo global (simula temperatura de luz del escaner)
    synthetic_rgb = np.clip(
        synthetic_rgb.astype(np.float64) * brightness_scale, 0, 255
    ).astype(np.uint8)

    return synthetic_rgb


# =============================================================================
# 5. GENERADOR DE VARIACIONES SINTETICAS
# =============================================================================

def generate_stain_variants(image_rgb, n_variants=5, seed=None):
    """
    Genera N variaciones sinteticas de una imagen HER2 con tincion aleatoria.
    
    Retorna:
        Lista de imagenes RGB sinteticas.
    """
    if seed is not None:
        np.random.seed(seed)

    variants = []
    for _ in range(n_variants):
        h_scale       = np.random.uniform(0.7, 1.3)
        dab_scale     = np.random.uniform(0.7, 1.3)
        h_shift       = np.random.uniform(-0.03, 0.03)
        dab_shift     = np.random.uniform(-0.03, 0.03)
        brightness    = np.random.uniform(0.85, 1.15)

        variant = augment_staining(
            image_rgb,
            h_scale=h_scale,
            dab_scale=dab_scale,
            h_shift=h_shift,
            dab_shift=dab_shift,
            brightness_scale=brightness
        )
        variants.append(variant)

    return variants


# =============================================================================
# 6. VISUALIZADOR DE CANALES SEPARADOS
# =============================================================================

def visualize_stain_separation(image_rgb, output_path=None):
    """
    Visualiza la imagen original y sus canales H y DAB separados.
    """
    concentrations, _ = separate_stains(image_rgb)

    # Reconstruir canal H puro (sin DAB)
    c_h = concentrations.copy()
    c_h[:, :, 1] = 0
    h_image = reconstruct_from_stains(c_h)

    # Reconstruir canal DAB puro (sin H)
    c_dab = concentrations.copy()
    c_dab[:, :, 0] = 0
    dab_image = reconstruct_from_stains(c_dab)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(h_image)
    axes[1].set_title("Canal H\n(Hematoxilina - Núcleos)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(dab_image)
    axes[2].set_title("Canal DAB\n(HER2 Membrana)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, facecolor='white')
        print(f"Separacion de canales guardada en: {output_path}")
    plt.show()


# =============================================================================
# 7. PRUEBA CON IMAGEN REAL
# =============================================================================

def run_test(image_path, output_dir):
    """
    Corre el pipeline completo de Capa 2 sobre una imagen JPEG o PNG real.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Cargando imagen: {image_path}")
    image_pil = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image_pil)

    print(f"Tamano: {image_rgb.shape}, dtype: {image_rgb.dtype}")

    # A. Separacion de canales
    visualize_stain_separation(
        image_rgb,
        output_path=os.path.join(output_dir, "separacion_canales.png")
    )

    # B. Generacion de 5 variaciones sinteticas
    print("Generando variaciones sinteticas de tincion...")
    variants = generate_stain_variants(image_rgb, n_variants=5, seed=42)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(image_rgb)
    axes[0].set_title("ORIGINAL", fontsize=12, fontweight='bold')
    axes[0].axis("off")

    for i, variant in enumerate(variants):
        axes[i+1].imshow(variant)
        axes[i+1].set_title(f"Sintética #{i+1}", fontsize=11)
        axes[i+1].axis("off")

    plt.suptitle("Augmentación de Tinción: Original vs Variaciones Sintéticas (Macenko)", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "variaciones_sinteticas.png")
    plt.savefig(out_path, dpi=200, facecolor='white')
    print(f"Panel de variaciones guardado en: {out_path}")
    plt.show()


if __name__ == "__main__":
    import sys

    # Uso: python macenko.py ruta/imagen.jpg
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Busca automaticamente una imagen en imagenes_test/
        base = os.path.join(os.path.dirname(__file__), "..", "imagenes_test")
        candidates = []
        if os.path.exists(base):
            for f in os.listdir(base):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".tif")):
                    candidates.append(os.path.join(base, f))

        if not candidates:
            print("No se encontraron imagenes en 'imagenes_test/'. Cargalas y vuelve a correr.")
            print("Uso: python macenko.py ruta/a/imagen.jpg")
            exit(1)

        img_path = candidates[0]
        print(f"Imagen detectada automaticamente: {img_path}")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output_sintetico")
    run_test(img_path, output_dir)
