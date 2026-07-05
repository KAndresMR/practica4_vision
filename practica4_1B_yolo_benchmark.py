#!/usr/bin/env python3
"""
Práctica 4 - Parte 1B: Pruebas de rendimiento YOLO CPU vs GPU
Asignatura: Visión por Computador

Este script ejecuta inferencia sobre un video con dos redes neuronales
(ej. YOLOv11/v12) comparando CPU y GPU.
Muestra en pantalla: FPS, Uso RAM, Uso VRAM (nvidia-smi), MAC Address.
"""

import cv2
import time
import argparse
import numpy as np
import psutil
import subprocess
import uuid
import os

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("Por favor instala ultralytics: pip install ultralytics torch")
    exit(1)

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

def process_video(video_path, model_name, device, output_path):
    print(f"\n--- Iniciando benchmark con {model_name} en {device.upper()} ---")
    
    # Cargar modelo (lo descargará automáticamente si no existe)
    print(f"Cargando modelo {model_name}...")
    model = YOLO(model_name)
    
    # Determinar device para ultralytics
    yolo_device = 0 if device == "gpu" and torch.cuda.is_available() else "cpu"
    device_label = "GPU (CUDA)" if yolo_device != "cpu" else "CPU"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error abriendo video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))
    
    mac_address = get_mac_address()
    fps_history = []
    frames_processed = 0
    
    print(f"Procesando video... (Presiona 'q' para detener)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        t_start = time.perf_counter()
        results = model.predict(source=frame, device=yolo_device, verbose=False)
        t_end = time.perf_counter()
        
        inf_time = t_end - t_start
        current_fps = 1.0 / inf_time if inf_time > 0 else 0
        fps_history.append(current_fps)
        
        annotated = results[0].plot()
        
        # --- OVERLAY ---
        # Fondo oscuro para el texto
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (450, 210), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color_val = (0, 255, 0) if yolo_device != "cpu" else (0, 128, 255)
        
        cv2.putText(annotated, f"Modelo: {model_name}", (20, 40), font, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated, f"Device: {device_label}", (20, 70), font, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (20, 100), font, 0.8, color_val, 2)
        
        ram_usage = psutil.Process().memory_info().rss / 1e6
        cv2.putText(annotated, f"RAM: {ram_usage:.0f} MB", (20, 130), font, 0.6, (255, 255, 255), 2)
        
        if yolo_device != "cpu":
            vram = get_gpu_memory_usage()
            cv2.putText(annotated, f"VRAM: {vram}", (20, 160), font, 0.6, (255, 255, 255), 2)
            
        cv2.putText(annotated, f"MAC: {mac_address}", (20, 190), font, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("YOLO Benchmark", annotated)
        out.write(annotated)
        frames_processed += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    avg_fps = np.mean(fps_history) if fps_history else 0
    print(f"Completado! Promedio {avg_fps:.1f} FPS. Guardado en {output_path}")
    return avg_fps


def main():
    parser = argparse.ArgumentParser(description="YOLO GPU vs CPU Benchmark")
    parser.add_argument("--video", type=str, required=True, help="Ruta al video de entrada")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: No se encontró el video {args.video}")
        return

    # Puedes cambiar los modelos según lo que hayan visto en clase (ej: yolo11n.pt, yolov8n.pt)
    model_1 = "yolo11n.pt" # Representa la red actual
    model_2 = "yolov8n.pt" # Representa la red anterior (o v26 si usaron algún fork específico)
    
    print("\n" + "="*50)
    print("  INICIANDO BENCHMARK PARTE 1B - YOLO")
    print("="*50)
    
    # 1. Benchmark CPU
    fps_m1_cpu = process_video(args.video, model_1, "cpu", f"out_{model_1}_cpu.mp4")
    fps_m2_cpu = process_video(args.video, model_2, "cpu", f"out_{model_2}_cpu.mp4")
    
    # 2. Benchmark GPU
    fps_m1_gpu = 0
    fps_m2_gpu = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        fps_m1_gpu = process_video(args.video, model_1, "gpu", f"out_{model_1}_gpu.mp4")
        torch.cuda.empty_cache()
        fps_m2_gpu = process_video(args.video, model_2, "gpu", f"out_{model_2}_gpu.mp4")
    else:
        print("\n⚠️ CUDA NO ESTÁ DISPONIBLE. Saltando pruebas de GPU.")
        
    print("\n" + "="*50)
    print("  RESULTADOS FINALES YOLO")
    print("="*50)
    print(f"{model_1:15} | CPU: {fps_m1_cpu:5.1f} FPS | GPU: {fps_m1_gpu:5.1f} FPS")
    print(f"{model_2:15} | CPU: {fps_m2_cpu:5.1f} FPS | GPU: {fps_m2_gpu:5.1f} FPS")
    print("="*50)

if __name__ == "__main__":
    main()
