
#!/usr/bin/env python3
"""
Práctica 4 - Parte 1A: Inferencia en tiempo real con webcam
Segmentación de frutas (Apple, Banana, Orange) con YOLOv11

Ejecutar en el laboratorio de Cómputo 8 con GPU NVIDIA:
    python3 webcam_fruit_seg.py --device gpu
    python3 webcam_fruit_seg.py --device cpu

Autor: [Tu Nombre]
Asignatura: Visión por Computador
"""

import cv2
import time
import argparse
import numpy as np
import psutil
import subprocess
import uuid

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("Instalar dependencias: pip install ultralytics torch")
    exit(1)


def get_mac_address():
    """Obtiene la MAC Address del computador."""
    mac = uuid.getnode()
    mac_str = ':'.join(f'{(mac >> (8 * i)) & 0xff:02x}' for i in reversed(range(6)))
    return mac_str


def get_gpu_memory_usage():
    """Obtiene el uso de memoria GPU con nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        used, total = result.stdout.strip().split(",")
        return f"{used.strip()} / {total.strip()} MB"
    except Exception:
        return "N/A"


def main():
    parser = argparse.ArgumentParser(description="Fruit Segmentation - Webcam")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="Dispositivo para inferencia: gpu o cpu")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="Ruta al modelo entrenado (.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Umbral de confianza")
    parser.add_argument("--camera", type=int, default=0,
                        help="Índice de la cámara (0 = default)")
    args = parser.parse_args()

    device = 0 if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    device_name = "GPU (CUDA)" if device != "cpu" else "CPU"

    print(f"
{'='*60}")
    print(f"  Práctica 4 - Segmentación de Frutas")
    print(f"  Dispositivo: {device_name}")
    print(f"  MAC Address: {get_mac_address()}")
    if device != "cpu":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {get_gpu_memory_usage()}")
    print(f"{'='*60}
")

    # Cargar modelo
    model = YOLO(args.model)
    print(f"Modelo cargado: {args.model}")

    # Abrir webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_history = []
    mac_address = get_mac_address()

    print("Presiona 'q' para salir, 's' para guardar screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia
        t_start = time.perf_counter()
        results = model.predict(source=frame, device=device, verbose=False, conf=args.conf)
        t_end = time.perf_counter()

        inference_time = t_end - t_start
        current_fps = 1.0 / inference_time if inference_time > 0 else 0
        fps_history.append(current_fps)

        # Dibujar resultado
        annotated = results[0].plot()

        # Overlay de información
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (420, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color_val = (0, 255, 0) if device != "cpu" else (0, 128, 255)

        cv2.putText(annotated, f"Device: {device_name}", (20, 35), font, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (20, 65), font, 0.8, color_val, 2)
        cv2.putText(annotated, f"Inference: {inference_time*1000:.1f} ms", (20, 95), font, 0.6, color_val, 2)

        ram_usage = psutil.Process().memory_info().rss / 1e6
        cv2.putText(annotated, f"RAM: {ram_usage:.0f} MB", (20, 125), font, 0.6, (255, 255, 255), 2)

        if device != "cpu":
            gpu_mem = get_gpu_memory_usage()
            cv2.putText(annotated, f"VRAM: {gpu_mem}", (20, 155), font, 0.6, (255, 255, 255), 2)

        cv2.putText(annotated, f"MAC: {mac_address}", (20, 175 if device != "cpu" else 155),
                    font, 0.5, (200, 200, 200), 1)

        n_dets = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated, f"Detections: {n_dets}",
                    (annotated.shape[1] - 200, 35), font, 0.6, (255, 255, 0), 2)

        cv2.imshow(f"Fruit Segmentation - {device_name}", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"screenshot_{device_name.lower().replace(' ', '_')}_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(filename, annotated)
            print(f"Screenshot guardado: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    # Resumen final
    if fps_history:
        print(f"
{'='*60}")
        print(f"  RESUMEN DE SESIÓN")
        print(f"  Dispositivo: {device_name}")
        print(f"  Frames procesados: {len(fps_history)}")
        print(f"  FPS promedio: {np.mean(fps_history):.2f}")
        print(f"  FPS mínimo:   {np.min(fps_history):.2f}")
        print(f"  FPS máximo:   {np.max(fps_history):.2f}")
        print(f"  MAC Address:  {mac_address}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
