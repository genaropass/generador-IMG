import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importar Macenko para la deteccion de Hematoxilina
import sys
sys.path.insert(0, os.path.dirname(__file__))
from macenko import separate_stains

class WSIScanner:
    def __init__(self, window_size=512, overlap_ratio=0.25):
        """
        Escáner de Whole Slide Images por Sliding Window.
        
        Args:
            window_size: Tamaño del parche a recortar (cuadrado).
            overlap_ratio: Porcentaje de superposición (0.0 a 1.0) para evitar cortar células.
        """
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.stride = int(window_size * (1.0 - overlap_ratio))

    def get_windows(self, image_shape):
        """
        Genera las coordenadas (x_start, y_start, x_end, y_end) para el escaneo.
        """
        H, W = image_shape[:2]
        windows = []
        
        for y in range(0, H - self.window_size + self.stride, self.stride):
            for x in range(0, W - self.window_size + self.stride, self.stride):
                y_end = min(y + self.window_size, H)
                x_end = min(x + self.window_size, W)
                y_start = max(0, y_end - self.window_size)
                x_start = max(0, x_end - self.window_size)
                windows.append((x_start, y_start, x_end, y_end))
        
        # Caso de imagenes muy pequeñas, agregar al menos una ventana
        if not windows:
            windows.append((0, 0, min(self.window_size, W), min(self.window_size, H)))
            
        # Remover duplicados (por los bordes)
        windows = list(set(windows))
        return windows


class HybridDetector:
    def __init__(self, prob_threshold=0.7):
        """
        Deteccion hibrida guiada por restricciones biologicas y fractales.
        Filtra candidatos (células) en base a su núcleo, textura, densidad y color.
        """
        self.prob_threshold = prob_threshold
        
    def _color_density_score(self, image_rgb):
        """
        Evalua la densidad de color. Tejidos casi blancos devuelven puntaje cercano a 0.
        """
        grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        # Porcentaje de pixeles que no son completamente blancos (> 220)
        tissue_mask = grayscale < 220
        density = np.sum(tissue_mask) / (image_rgb.shape[0] * image_rgb.shape[1])
        return min(density * 1.5, 1.0)  # Escalar un poco para dar margen

    def _texture_score(self, image_rgb):
        """
        Evalua la varianza local (textura) para descartar ruido plano.
        """
        grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        variance = np.var(grayscale)
        # Si la varianza es muy baja, es fondo o artefacto
        score = min(variance / 1000.0, 1.0)
        return score
        
    def detect_candidates(self, window_rgb, global_offset_x=0, global_offset_y=0):
        """
        Detecta células candidatas dentro de un parche (ventana).
        Retorna: Lista de diccionarios con {centroid, bbox, prob_score}.
        """
        H, W = window_rgb.shape[:2]
        
        # 1. Filtro rapido de densidad para saltar parches vacios (Acelera el pipeline)
        density_score = self._color_density_score(window_rgb)
        if density_score < 0.1:
            return [] # Parche casi vacio
            
        texture_score = self._texture_score(window_rgb)
            
        # 2. Separacion Macenko para obtener Hematoxilina (Nucleos)
        concentrations, _ = separate_stains(window_rgb)
        h_channel = concentrations[:, :, 0] # Hematoxilina pura
        
        # 3. Deteccion de nucleos (Threshold)
        # Normalizar h_channel para thresholding (0-255)
        h_norm = cv2.normalize(h_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Umbral adaptativo (Otsu)
        _, thresh = cv2.threshold(h_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 4. Componentes Conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        candidates = []
        for i in range(1, num_labels): # Ignorar el 0 (fondo)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Etapa 3 del Blueprint: Validación Morfológica del Candidato
            # Descartar polvo (area muy chica) o manchas gigantes (area inmensa)
            if area < 50 or area > 3000:
                continue
                
            x_left = stats[i, cv2.CC_STAT_LEFT]
            y_top = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calcular Circularidad aprox del nucleo
            perimeter = 2 * (width + height) # rough estimate
            circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
            if circularity < 0.2: # Descartar fibras alargadas (no son nucleos)
                continue
                
            cx, cy = centroids[i]
            
            # Calcular Probabilidad Híbrida del candidato
            # P = 0.45*Nucleo + 0.30*Textura + 0.15*Densidad + 0.10*Color (Aproximacion)
            nucleus_score = min(area / 500.0, 1.0)
            
            # Color score basado en intensidad DAB local (HER2)
            dab_channel = concentrations[y_top:y_top+height, x_left:x_left+width, 1]
            color_score = min(np.mean(dab_channel) * 2.0, 1.0) if dab_channel.size > 0 else 0
            
            prob_score = (0.45 * nucleus_score) + (0.30 * texture_score) + (0.15 * density_score) + (0.10 * color_score)
            
            if prob_score > self.prob_threshold:
                # El Bounding Box del núcleo se expande matemáticamente para incluir la membrana 
                # (ya que el nucleo es solo el centro). Multiplicamos por un factor.
                expand_factor = 1.5
                nx_left = max(0, int(cx - (width/2) * expand_factor))
                ny_top = max(0, int(cy - (height/2) * expand_factor))
                nx_right = min(W, int(cx + (width/2) * expand_factor))
                ny_bottom = min(H, int(cy + (height/2) * expand_factor))
                
                bbox = [nx_left, ny_top, nx_right, ny_bottom]
                
                # Coordenadas Globales (en la WSI)
                global_cx = cx + global_offset_x
                global_cy = cy + global_offset_y
                global_bbox = [
                    nx_left + global_offset_x, 
                    ny_top + global_offset_y, 
                    nx_right + global_offset_x, 
                    ny_bottom + global_offset_y
                ]
                
                candidates.append({
                    "local_centroid": (cx, cy),
                    "local_bbox": bbox,
                    "global_centroid": (global_cx, global_cy),
                    "global_bbox": global_bbox,
                    "prob_score": prob_score,
                    "area": area
                })
                
        return candidates


def run_stage_1_and_2(image_path, output_dir):
    """
    Ejecuta la Etapa 1 (WSI Scanning) y Etapa 2 (Localizacion Hibrida) del Blueprint.
    Genera un panel visual para verificar cientificamente los candidatos detectados.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{'Etapa 1':^10}] Leyendo Whole Slide Image: {os.path.basename(image_path)}")
    
    img_pil = Image.open(image_path).convert("RGB")
    wsi_rgb = np.array(img_pil)
    
    # Iniciar Escaner y Detector
    scanner = WSIScanner(window_size=256, overlap_ratio=0.2)
    windows = scanner.get_windows(wsi_rgb.shape)
    
    detector = HybridDetector(prob_threshold=0.6) # Bajamos un poco para capturar HER2 debiles
    
    all_candidates = []
    
    print(f"[{'Etapa 1':^10}] Sliding Window genero {len(windows)} ventanas.")
    print(f"[{'Etapa 2':^10}] Buscando celulas con Heatmap Híbrido...")
    
    for (x1, y1, x2, y2) in windows:
        window_rgb = wsi_rgb[y1:y2, x1:x2]
        
        # Detectar en esta ventana
        candidates = detector.detect_candidates(window_rgb, global_offset_x=x1, global_offset_y=y1)
        all_candidates.extend(candidates)
        
    print(f"[{'Etapa 2':^10}] Finalizado. Se encontraron {len(all_candidates)} candidatos robustos.")
    
    # Remover candidatos duplicados por el overlap de ventanas
    # Usando NMS (Non-Maximum Suppression) simple basado en distancia de centroides
    filtered_candidates = []
    for c1 in sorted(all_candidates, key=lambda x: x["prob_score"], reverse=True):
        duplicate = False
        for c2 in filtered_candidates:
            # Distancia euclidiana entre centroides
            dist = np.sqrt((c1["global_centroid"][0] - c2["global_centroid"][0])**2 + 
                           (c1["global_centroid"][1] - c2["global_centroid"][1])**2)
            if dist < 20: # Si estan a menos de 20px, es la misma celula
                duplicate = True
                break
        if not duplicate:
            filtered_candidates.append(c1)
            
    print(f"[{'Validación':^10}] Filtro de Overlap: {len(filtered_candidates)} celulas unicas retenidas.")
    
    # -------------------------------------------------------------------------
    # Visualizacion Cientifica (Verificación)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(wsi_rgb)
    ax.set_title(f"Etapa 1 y 2 - DigPatho\n{len(filtered_candidates)} Células detectadas por Densidad y Macenko", fontsize=14)
    
    # Dibujar grid de ventanas
    for (x1, y1, x2, y2) in windows:
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='white', linestyle=':', alpha=0.3)
        ax.add_patch(rect)
        
    # Dibujar Bounding Boxes y Centroides de Candidatos
    for cand in filtered_candidates:
        cx, cy = cand["global_centroid"]
        bx1, by1, bx2, by2 = cand["global_bbox"]
        
        # Bounding box
        rect = plt.Rectangle((bx1, by1), bx2-bx1, by2-by1, fill=False, edgecolor='lime', linewidth=1.5)
        ax.add_patch(rect)
        
        # Centroide (Prompt Point para SAM 2)
        ax.plot(cx, cy, 'ro', markersize=3)
        
        # Score
        ax.text(bx1, by1-2, f"{cand['prob_score']:.2f}", color='lime', fontsize=8)
        
    ax.axis("off")
    out_path = os.path.join(output_dir, "blueprint_etapa1_2.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"[{'Salida':^10}] Mapa de Deteccion guardado en: {out_path}")
    plt.close(fig)
    
    return filtered_candidates, windows, wsi_rgb


if __name__ == "__main__":
    IMAGE_PATH = r"E:\Genaro\Desktop\Digital pathologies\no entrar\scripts\3+\3+\3Her2.jpg"
    OUTPUT_DIR = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico"
    run_stage_1_and_2(IMAGE_PATH, OUTPUT_DIR)
