#!/usr/bin/env python3
"""
Práctica 4 - Parte 1B: Pruebas de rendimiento Super Resolución CPU vs GPU
Asignatura: Visión por Computador

Requisito: Ejecutar una red de Super Resolución (ej. Real-ESRGAN) en CPU vs GPU.
Dependencias extra: pip install basicsr facexlib gfpgan realesrgan
"""

import cv2
import time
import argparse
import numpy as np
import psutil
import subprocess
import uuid
import os
import urllib.request

try:
    import torch
except ImportError:
    print("Por favor instala torch: pip install torch torchvision")
    exit(1)

# Intentar importar Real-ESRGAN
HAS_REAL_ESRGAN = False
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    HAS_REAL_ESRGAN = True
except ImportError:
    print("\n⚠️ ADVERTENCIA: No se encontró la librería 'realesrgan'.")
    print("Para usar Real-ESRGAN real instala: pip install basicsr facexlib gfpgan realesrgan")
    print("El script continuará usando un modelo dummy en PyTorch (Bicúbico) para demostrar el pipeline y medir la GPU.\n")


def get_mac_address():
    mac = uuid.getnode()
    return ':'.join(f'{(mac >> (8 * i)) & 0xff:02x}' for i in reversed(range(6)))

def get_gpu_memory_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        used, total = result.stdout.strip().split(",")
        return f"{used.strip()} / {total.strip()} MB"
    except Exception:
        return "N/A"

def download_model():
    model_path = 'RealESRGAN_x4plus_anime_6B.pth'
    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    if not os.path.exists(model_path) and HAS_REAL_ESRGAN:
        print(f"Descargando modelo Real-ESRGAN ({url})...")
        urllib.request.urlretrieve(url, model_path)
        print("Descarga completa.")
    return model_path


def process_video_sr(video_path, device, output_path):
    print(f"\n--- Iniciando Super Resolución en {device.upper()} ---")
    
    torch_device = torch.device('cuda' if device == 'gpu' and torch.cuda.is_available() else 'cpu')
    device_label = "GPU (CUDA)" if torch_device.type == 'cuda' else "CPU"
    
    # Inicializar modelo
    if HAS_REAL_ESRGAN:
        model_path = download_model()
        # Usar un modelo más pequeño para que no crashee por VRAM
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4, 
            model_path=model_path, 
            model=model, 
            tile=400, # Tiling para ahorrar VRAM
            tile_pad=10, 
            pre_pad=0, 
            half=True if torch_device.type == 'cuda' else False,
            device=torch_device
        )
        model_name = "Real-ESRGAN x4"
    else:
        # Fallback dummy si no está instalada la librería
        model_name = "PyTorch Bicubic (Dummy SR)"
        upsampler = None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error abriendo video {video_path}")
        return

    # Escalar la resolución de salida
    scale = 4
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))
    
    mac_address = get_mac_address()
    fps_history = []
    
    print(f"Procesando video... (Presiona 'q' para detener)")
    # Leer solo 50 frames para que no demore horas (SR es muy lento)
    max_frames = 50 
    frames_processed = 0
    
    while cap.isOpened() and frames_processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        t_start = time.perf_counter()
        
        if HAS_REAL_ESRGAN:
            # Real-ESRGAN Inferencia
            output_img, _ = upsampler.enhance(frame, outscale=scale)
        else:
            # Fallback: Hacer Resize en GPU usando PyTorch para simular carga
            tensor_img = torch.from_numpy(frame).to(torch_device).float()
            tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0)
            upsampled = torch.nn.functional.interpolate(tensor_img, scale_factor=scale, mode='bicubic', align_corners=False)
            output_img = upsampled.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
            
        t_end = time.perf_counter()
        
        inf_time = t_end - t_start
        current_fps = 1.0 / inf_time if inf_time > 0 else 0
        fps_history.append(current_fps)
        
        # --- OVERLAY INFO ---
        annotated = output_img.copy()
        
        # Como la imagen ahora es x4 más grande, escalamos la fuente
        font_scale = 1.5
        th = 3
        
        overlay = annotated.copy()
        cv2.rectangle(overlay, (20, 20), (900, 350), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color_val = (0, 255, 0) if torch_device.type != "cpu" else (0, 128, 255)
        
        cv2.putText(annotated, f"Modelo: {model_name}", (40, 80), font, font_scale, (255, 255, 0), th)
        cv2.putText(annotated, f"Device: {device_label}", (40, 140), font, font_scale, (0, 255, 255), th)
        cv2.putText(annotated, f"FPS: {current_fps:.2f} ({inf_time*1000:.0f} ms)", (40, 200), font, font_scale, color_val, th)
        
        ram_usage = psutil.Process().memory_info().rss / 1e6
        cv2.putText(annotated, f"RAM: {ram_usage:.0f} MB", (40, 260), font, font_scale, (255, 255, 255), th)
        
        if torch_device.type != "cpu":
            vram = get_gpu_memory_usage()
            cv2.putText(annotated, f"VRAM: {vram}", (40, 320), font, font_scale, (255, 255, 255), th)
            
        cv2.putText(annotated, f"MAC: {mac_address}", (950, 80), font, 1.0, (200, 200, 200), 2)
        
        # Mostrar versión redimensionada para que quepa en pantalla
        display_img = cv2.resize(annotated, (int(width/2), int(height/2)))
        cv2.imshow("Super Resolucion Benchmark", display_img)
        
        out.write(annotated)
        frames_processed += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    avg_fps = np.mean(fps_history) if fps_history else 0
    print(f"Completado! Promedio {avg_fps:.2f} FPS. Guardado en {output_path}")
    return avg_fps


def main():
    parser = argparse.ArgumentParser(description="Super Resolucion GPU vs CPU Benchmark")
    parser.add_argument("--video", type=str, required=True, help="Ruta al video de entrada (recomendado video de baja resolución 360p)")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: No se encontró el video {args.video}")
        return

    print("\n" + "="*50)
    print("  INICIANDO BENCHMARK PARTE 1B - SUPER RESOLUCION")
    print("="*50)
    
    # 1. Benchmark CPU
    fps_cpu = process_video_sr(args.video, "cpu", "out_sr_cpu.mp4")
    
    # 2. Benchmark GPU
    fps_gpu = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        fps_gpu = process_video_sr(args.video, "gpu", "out_sr_gpu.mp4")
    else:
        print("\n⚠️ CUDA NO ESTÁ DISPONIBLE. Saltando pruebas de GPU.")
        
    print("\n" + "="*50)
    print("  RESULTADOS FINALES SUPER RESOLUCION")
    print("="*50)
    print(f"CPU: {fps_cpu:5.2f} FPS | GPU: {fps_gpu:5.2f} FPS")
    print(f"Mejora de velocidad (Speedup): {fps_gpu/fps_cpu if fps_cpu > 0 else 0:.1f}x")
    print("="*50)

if __name__ == "__main__":
    main()
