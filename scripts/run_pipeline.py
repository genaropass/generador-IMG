"""
DigPatho – Run Guide
Script de inicio rapido para ejecutar el pipeline completo.
"""
import os, sys

# Verificar que estamos en el venv correcto
venv_python = r"E:\Genaro\Desktop\Digital pathologies\venv\Scripts\python.exe"
if not os.path.exists(venv_python):
    print("[ERROR] No se encontro el venv. Crea el entorno primero:")
    print("  python -m venv venv")
    print("  venv\\Scripts\\pip install ultralytics opencv-python scipy scikit-image")
    sys.exit(1)

import subprocess, argparse

SCRIPTS = r"E:\Genaro\Desktop\Digital pathologies\no entrar\scripts"
OUTPUT  = r"E:\Genaro\Desktop\Digital pathologies\no entrar\output_sintetico"

def run(script, *args):
    cmd = [venv_python, os.path.join(SCRIPTS, script)] + list(args)
    print(f"\n>> {script}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DigPatho – Pipeline Sintetico")
    parser.add_argument("--image",   required=True, help="Ruta a la imagen fuente (JPEG/PNG)")
    parser.add_argument("--grade",   default="3+",  help="Grado HER2 (0, 1+, 2+, 3+)")
    parser.add_argument("--variants",default="3",   help="Variantes sinteticas por celula")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Imagen no encontrada: {args.image}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  DigPatho – Pipeline de Generacion Sintetica")
    print(f"  Imagen : {args.image}")
    print(f"  Grado  : {args.grade}")
    print(f"  Output : {OUTPUT}")
    print(f"{'='*55}")

    # Etapa 1-2: Detectar candidatos
    run("wsi_scanner.py")

    # Etapas 3-5: SAM2 + Extraccion
    run("sam_segmentation.py")

    # Etapa 6: Caracterizacion fractal
    run("fractal_characterizer.py")

    # Etapas 7-9: Generar dataset
    run("generate_dataset.py")

    # Generar comparativa WSI
    run("comparativa_wsi.py")

    # Exportar informe unificado
    run("exportar_informe_completo.py")

    print(f"\n{'='*55}")
    print(f"  Pipeline completado.")
    print(f"  Dataset: {OUTPUT}\\dataset_sintetico")
    print(f"  Informe: E:\\Genaro\\Desktop\\Digital pathologies\\no entrar\\DigPatho_Pipeline_Completo.html")
    print(f"{'='*55}\n")
