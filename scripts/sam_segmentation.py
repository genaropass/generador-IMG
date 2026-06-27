import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from ultralytics import SAM

# Importar el escáner de la etapa anterior
import sys
sys.path.insert(0, os.path.dirname(__file__))
from wsi_scanner import run_stage_1_and_2

class Stage345Processor:
    def __init__(self, model_name="sam2.1_t.pt"):
        print(f"[{'Etapa 3':^10}] Cargando modelo fundacional {model_name}...")
        self.model = SAM(model_name)
        
    def process_candidates(self, wsi_rgb, candidates):
        """
        Itera sobre los candidatos, refina con SAM2 y extrae la celula RGBA validada.
        """
        valid_cells = []
        H_wsi, W_wsi = wsi_rgb.shape[:2]
        
        print(f"[{'Etapa 3':^10}] Iniciando segmentacion focalizada de {len(candidates)} candidatos...")
        
        for idx, cand in enumerate(candidates):
            gcx, gcy = cand["global_centroid"]
            gbx1, gby1, gbx2, gby2 = cand["global_bbox"]
            
            # Ampliar el bounding box original para darle contexto a SAM, 
            # pero sin pasarle toda la imagen gigante.
            padding = 40
            px1 = max(0, int(gbx1) - padding)
            py1 = max(0, int(gby1) - padding)
            px2 = min(W_wsi, int(gbx2) + padding)
            py2 = min(H_wsi, int(gby2) + padding)
            
            # Recorte contextual (Imagen reducida para SAM)
            patch_rgb = wsi_rgb[py1:py2, px1:px2]
            
            # Coordenadas relativas al parche
            local_cx = gcx - px1
            local_cy = gcy - py1
            local_bx1 = gbx1 - px1
            local_by1 = gby1 - py1
            local_bx2 = gbx2 - px1
            local_by2 = gby2 - py1
            
            # ETAPA 3: Inyeccion a SAM 2 (Image + Point + BBox)
            # Segun el Blueprint: SAM recibe TODO el contexto necesario para entender la membrana
            results = self.model.predict(
                source=patch_rgb,
                points=[[local_cx, local_cy]],
                bboxes=[[local_bx1, local_by1, local_bx2, local_by2]],
                labels=[1],
                verbose=False
            )
            
            if not results or results[0].masks is None or len(results[0].masks.data) == 0:
                continue
                
            mask = results[0].masks.data[0].cpu().numpy()
            
            # Ajustar resolucion de la mascara al tamano del parche si difiere
            ph, pw = patch_rgb.shape[:2]
            if mask.shape != (ph, pw):
                mask = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)
                
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Obtener el contorno principal
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            main_contour = max(contours, key=cv2.contourArea)
            
            # ETAPA 4: Validacion de la Segmentacion (Filtros Biologicos)
            area = cv2.contourArea(main_contour)
            if area < 200 or area > 8000:
                continue # Falla validacion de area biologica (polvo o cluster gigante)
                
            perimeter = cv2.arcLength(main_contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
            if circularity < 0.15:
                continue # Falla circularidad (artefacto muy alargado, ej: fibra de colageno)
                
            # Validacion Edge-Touch (si la mascara toca el borde del patch_rgb, es un corte artificial)
            x, y, w, h = cv2.boundingRect(main_contour)
            margin = 2
            if x <= margin or y <= margin or (x + w) >= (pw - margin) or (y + h) >= (ph - margin):
                continue # Edge touch fail -> Descartar celula truncada
                
            # ETAPA 5: Extraccion (Isolacion de Célula)
            # Crear imagen RGBA
            rgba = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = mask_uint8 # Alpha channel = SAM mask
            
            # Recortar la celula aislada ajustada a su limite exacto
            isolated_cell = rgba[y:y+h, x:x+w]
            isolated_mask = mask_uint8[y:y+h, x:x+w]
            
            # Contorno local para la etapa posterior (Etapa 6 Fractal)
            rel_contours, _ = cv2.findContours(isolated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not rel_contours:
                continue
            rel_contour = max(rel_contours, key=cv2.contourArea)
            
            valid_cells.append({
                "id": f"cell_{idx:04d}",
                "rgba": isolated_cell,
                "mask": isolated_mask,
                "contour": rel_contour,
                "global_x": px1 + x,
                "global_y": py1 + y,
                "area": area,
                "circularity": circularity
            })
            
        print(f"[{'Etapa 4':^10}] Validacion matematica completada. {len(valid_cells)} células puras superaron todos los umbrales.")
        return valid_cells


def save_extracted_cells(cells, output_dir):
    """
    Exporta las celulas como entidades independientes.
    """
    cells_dir = os.path.join(output_dir, "extracted_cells")
    os.makedirs(cells_dir, exist_ok=True)
    
    coords = {}
    for cell in cells:
        # RGBA transparente
        img_path = os.path.join(cells_dir, f"{cell['id']}.png")
        Image.fromarray(cell['rgba']).save(img_path)
        # Mascara binaria
        mask_path = os.path.join(cells_dir, f"{cell['id']}_mask.png")
        Image.fromarray(cell['mask']).save(mask_path)
        
        coords[cell['id']] = {
            "global_x": cell["global_x"],
            "global_y": cell["global_y"]
        }
        
    import json
    with open(os.path.join(cells_dir, "cell_coords.json"), "w") as f:
        json.dump(coords, f)
        
    print(f"[{'Etapa 5':^10}] Extraccion completada. {len(cells)} muestras aisladas guardadas en: {cells_dir}")


def run_stages_3_to_5(image_path, output_dir):
    # 1. Obtener candidatos hibridos
    candidates, _, wsi_rgb = run_stage_1_and_2(image_path, output_dir)
    
    # 2. Inicializar procesador SAM
    processor = Stage345Processor()
    
    # 3. Procesar y Extraer
    cells = processor.process_candidates(wsi_rgb, candidates)
    
    # 4. Guardar resultados
    save_extracted_cells(cells, output_dir)
    
    # 5. Collage Visual de Verificacion
    if cells:
        n_show = min(16, len(cells))
        cols = 4
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        
        # Manejar array de axes si es 1D o 0D
        if rows == 1 and cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for i, ax in enumerate(axes):
            if i < n_show:
                # Mostrar RGBA sobre fondo negro para destacar la membrana detectada
                ax.imshow(cells[i]['rgba'])
                ax.set_title(f"A:{int(cells[i]['area'])} C:{cells[i]['circularity']:.2f}", fontsize=9, color='white')
            ax.set_facecolor('#111111')
            ax.axis('off')
            
        plt.tight_layout()
        collage_path = os.path.join(output_dir, "blueprint_etapa345.png")
        fig.patch.set_facecolor('#111111')
        plt.savefig(collage_path, dpi=200, facecolor=fig.get_facecolor())
        print(f"[{'Salida':^10}] Collage de extracción RGBA guardado en: {collage_path}")
        plt.close(fig)

if __name__ == "__main__":
    IMAGE_PATH = r"E:\Genaro\Desktop\Digital pathologies\no entrar\scripts\3+\3+\3Her2.jpg"
    OUTPUT_DIR = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico"
    run_stages_3_to_5(IMAGE_PATH, OUTPUT_DIR)
