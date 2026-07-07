import os
import sys
import argparse
import json
import csv
import uuid
import datetime
import hashlib
import gc
import time
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

# Importamos funciones del pipeline
from synthetic_pipeline import (
    detect_cell_candidates,
    segment_with_sam2,
    characterize_cells,
    generate_synthetic_wsi,
    save_variant
)
from macenko import calibrate_reference, HE_REFERENCE
try:
    from tqdm import tqdm
except ImportError:
    print("Por favor instala tqdm: pip install tqdm")
    sys.exit(1)

GENERATOR_VERSION = "v3.0"

def compute_sha256(filepath):
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def init_master_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "timestamp", "generator_version", 
                "original_file", "original_sha256", "synthetic_file",
                "grade", "stain_type", "alpha", "seed", 
                "resolution", "status", "processing_time_s", "error_message"
            ])

def log_to_master_csv(csv_path, data):
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def log_error(error_log_path, image_path, error_msg):
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}] ERROR in {image_path}: {error_msg}\n")

def process_single_image(image_path, output_class_dir, grade, stain_type, n_variants, alpha, master_csv, error_log):
    start_time = time.time()
    basename = os.path.basename(image_path)
    
    # Check if empty or invalid format
    try:
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        log_error(error_log, image_path, f"Error abriendo imagen: {str(e)}")
        return False
        
    H, W = image_rgb.shape[:2]
    if H == 0 or W == 0:
        log_error(error_log, image_path, "Resolucion invalida (0px)")
        return False
        
    if np.isnan(image_rgb).any():
        log_error(error_log, image_path, "La imagen original contiene NaNs")
        return False

    orig_sha256 = compute_sha256(image_path)
    
    try:
        candidates = detect_cell_candidates(image_rgb)
        if not candidates:
            raise ValueError("No se detectaron candidatos en la imagen.")
            
        masks_with_coords = segment_with_sam2(image_rgb, candidates)
        if not masks_with_coords:
             masks_with_coords = [(np.zeros((H, W), dtype=np.uint8), cx, cy) for (cx, cy) in candidates]
             
        characterized = characterize_cells(masks_with_coords)
        stain_ref = calibrate_reference(image_rgb) if stain_type == "ihc" else HE_REFERENCE
        
        for vi in range(n_variants):
            var_start = time.time()
            seed = 1000 + vi * 137
            uid = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()
            
            synth_rgb, new_coords, dx, dy = generate_synthetic_wsi(
                image_rgb, characterized, stain_ref, variant_idx=vi, grade=grade
            )
            
            if np.isnan(synth_rgb).any():
                raise ValueError(f"Variante {vi} generó NaNs")
                
            # Guardamos la variante en la subcarpeta
            sid, out_path = save_variant(
                output_class_dir, basename, vi, synth_rgb, new_coords, characterized, dx, dy, draw_points=False
            )
            
            proc_time = round(time.time() - var_start, 2)
            syn_filename = os.path.basename(out_path)
            
            log_to_master_csv(master_csv, [
                uid, timestamp, GENERATOR_VERSION,
                basename, orig_sha256, syn_filename,
                grade, stain_type, alpha, seed,
                f"{W}x{H}", "OK", proc_time, ""
            ])
            
    except Exception as e:
        log_error(error_log, image_path, str(e))
        uid = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        log_to_master_csv(master_csv, [
            uid, timestamp, GENERATOR_VERSION,
            basename, orig_sha256, "",
            grade, stain_type, alpha, "",
            f"{W}x{H}", "ERROR", round(time.time() - start_time, 2), str(e)
        ])
        return False
        
    finally:
        # Limpieza obligatoria de GPU/Memoria
        if 'image_rgb' in locals(): del image_rgb
        if 'synth_rgb' in locals(): del synth_rgb
        if 'masks_with_coords' in locals(): del masks_with_coords
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return True


def run_rapido(args):
    print(f"=== Modo Rapido ===")
    print(f"Imagen: {args.input} | Grado: {args.grade} | Variantes: {args.variants}")
    os.makedirs(args.output, exist_ok=True)
    out_dir = os.path.join(args.output, f"HER2_{args.grade}")
    os.makedirs(out_dir, exist_ok=True)
    
    master_csv = os.path.join(args.output, "dataset_metadata.csv")
    error_log = os.path.join(args.output, "error.log")
    init_master_csv(master_csv)
    
    process_single_image(
        args.input, out_dir, args.grade, args.stain, 
        args.variants, args.alpha, master_csv, error_log
    )
    print("Completado.")

def run_masivo(args):
    print(f"=== Modo Masivo ===")
    os.makedirs(args.output, exist_ok=True)
    master_csv = os.path.join(args.output, "dataset_metadata.csv")
    error_log = os.path.join(args.output, "error.log")
    checkpoint_file = os.path.join(args.output, "checkpoint.json")
    
    init_master_csv(master_csv)
    
    # Parse variants per class
    # Format: "HER2_0:10,HER2_1+:8,HER2_2+:15,HER2_3+:4"
    var_map = {}
    if args.variants_per_class:
        for pair in args.variants_per_class.split(','):
            cls, cnt = pair.split(':')
            var_map[cls.strip()] = int(cnt.strip())
            
    # Load checkpoint
    processed_images = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            cp = json.load(f)
            processed_images = set(cp.get("processed", []))
            
        if not args.resume:
            ans = input(f"Se encontro checkpoint con {len(processed_images)} imagenes procesadas. Continuar? [S/n]: ")
            if ans.lower() == 'n':
                processed_images = set()
    
    # Find all images
    tasks = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    for root, dirs, files in os.walk(args.input):
        folder_name = os.path.basename(root)
        if folder_name.startswith("HER2_"):
            grade = folder_name.replace("HER2_", "")
            n_vars = var_map.get(folder_name, 3) # default 3 if not specified
            
            for f in files:
                if f.lower().endswith(valid_exts):
                    img_path = os.path.join(root, f)
                    if img_path not in processed_images:
                        tasks.append((img_path, folder_name, grade, n_vars))
                        
    if not tasks:
        print("No hay nuevas imagenes para procesar.")
        return
        
    print(f"Total a procesar: {len(tasks)} imagenes.")
    
    for img_path, folder_name, grade, n_vars in tqdm(tasks, desc="Procesando Lote"):
        out_dir = os.path.join(args.output, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        
        process_single_image(
            img_path, out_dir, grade, args.stain, 
            n_vars, args.alpha, master_csv, error_log
        )
        
        # Update checkpoint
        processed_images.add(img_path)
        with open(checkpoint_file, "w") as f:
            json.dump({"processed": list(processed_images), "last_updated": datetime.datetime.now().isoformat()}, f)
            
    print("Procesamiento Masivo Completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Processor para DigPatho")
    parser.add_argument("--mode", required=True, choices=["rapido", "masivo"])
    parser.add_argument("--input", required=True, help="Ruta a imagen o carpeta raiz")
    parser.add_argument("--output", required=True, help="Carpeta de salida")
    parser.add_argument("--grade", help="Grado HER2 (solo modo rapido)")
    parser.add_argument("--stain", default="ihc", choices=["ihc", "he"], help="Tipo de tincion")
    parser.add_argument("--variants", type=int, default=3, help="Cant. de variantes (rapido)")
    parser.add_argument("--variants-per-class", help="Variantes por clase (ej: HER2_0:10,HER2_1+:8)")
    parser.add_argument("--alpha", type=float, default=0.85, help="Fuerza de deformacion")
    parser.add_argument("--resume", action="store_true", help="Continuar sin preguntar")
    
    args = parser.parse_args()
    
    if args.mode == "rapido":
        if not args.grade:
            print("ERROR: --grade es requerido en modo rapido.")
            sys.exit(1)
        run_rapido(args)
    elif args.mode == "masivo":
        run_masivo(args)
