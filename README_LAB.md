# Práctica 4 - Pruebas en Laboratorio Cómputo 8 (GPU)

Estos scripts están diseñados para cumplir con la **Parte 1B** de la rúbrica y las pruebas de la **Parte 1A** en el laboratorio de la UPS, utilizando las computadoras con tarjeta gráfica (GPU/CUDA).

## 1. Preparación del Entorno

Abre una terminal y asegúrate de tener las librerías necesarias instaladas:

```bash
pip install -r requirements_lab.txt
```

*(Si tienes problemas instalando `realesrgan`, el script de super resolución usará automáticamente un modelo alternativo para que puedas hacer la demostración de todas formas).*

---

## 2. Ejecutar Pruebas Parte 1A (Webcam Frutas)

Para esta prueba debes haber colocado el archivo `best.pt` (tu modelo entrenado en Colab) en la misma carpeta.

**Probar en GPU:**
```bash
python3 webcam_fruit_seg.py --device gpu --model best.pt
```

**Probar en CPU:**
```bash
python3 webcam_fruit_seg.py --device cpu --model best.pt
```

---

## 3. Ejecutar Pruebas Parte 1B (YOLO)

Para esta prueba necesitas un video corto grabado previamente (ej. `video_calle.mp4`). El script descargará automáticamente los modelos YOLO base (ej. YOLOv11) si no los tienes.

```bash
python3 practica4_1B_yolo_benchmark.py --video video_calle.mp4
```

> **Nota:** El script ejecutará automáticamente 2 modelos (yolo11n, yolov8n) en CPU y luego en GPU, generando videos de salida con el overlay de la información solicitada en la rúbrica (FPS, RAM, VRAM, MAC Address).

---

## 4. Ejecutar Pruebas Parte 1B (Super Resolución)

Para esta prueba necesitas un video (idealmente de resolución baja como 360p para evitar que la VRAM colapse al hacer el upscale x4).

```bash
python3 practica4_1B_superres_benchmark.py --video video_calle_360p.mp4
```

> **Nota:** El script ejecutará automáticamente la inferencia en CPU y GPU y mostrará la pantalla dividida. Al igual que YOLO, dejará guardados los videos `out_sr_cpu.mp4` y `out_sr_gpu.mp4`.

---

## 5. Evidencias para Entregar

Recuerda **grabar la pantalla** (usando OBS u otro software) mientras ejecutas las pruebas en la GPU. La rúbrica exige mostrar:
1. Terminal abierta (aunque el script ya pone la info en el video).
2. Cuadros por segundo (FPS) en GPU vs CPU.
3. Uso de memoria RAM.
4. MAC Address del computador (el script la imprime).

Los scripts ya muestran toda esta información directamente sobrepuesta en los videos de resultado para facilitar la demostración.
