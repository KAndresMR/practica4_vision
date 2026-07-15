# Práctica 4 - Pruebas en Laboratorio Cómputo 8 (GPU)

Estos scripts están diseñados para cumplir con las **3 partes** de la rúbrica de la Práctica 4, utilizando las computadoras con tarjeta gráfica (GPU/CUDA).

---

## 0. Preparación ANTES de ir al laboratorio

Debes llevar en un **pendrive** (o hacer `git pull` desde tu repo):
- Toda la carpeta `practica4/`
- Tu modelo `best.pt` (de la Parte 1A, ya está en `practica4_1A_export/`)
- Un **video corto (5-10 seg) en .mp4** de una calle o de objetos (para las pruebas 1B y 1C)

---

## 1. Configurar la Máquina (Apenas llegues)

```bash
# Instalar dependencias de Python
pip install -r requirements_lab.txt

# Verificar que CUDA está disponible
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### ⚠️ MUY IMPORTANTE: nvidia-smi en terminal separada

La rúbrica pide mostrar **el comando `nvidia-smi`** en la grabación. Abre una **segunda terminal** y ejecuta:

```bash
watch -n 1 nvidia-smi
```

Esto mostrará el uso de la GPU actualizándose cada segundo. **Deja esta terminal visible** mientras ejecutas todas las pruebas. Así quedará grabado en el video de evidencia.

---

## 2. Parte 1A — Webcam con Frutas (Segmentación)

Copia tu `best.pt` a esta carpeta (o usa la ruta completa).

**Probar en GPU:**
```bash
python3 practica4_1A_export/webcam_fruit_seg.py --device gpu --model practica4_1A_export/best.pt
```

**Probar en CPU:**
```bash
python3 practica4_1A_export/webcam_fruit_seg.py --device cpu --model practica4_1A_export/best.pt
```

- Pon una fruta (manzana, banana, naranja) frente a la webcam
- Observa los FPS, RAM, VRAM y MAC Address en el overlay
- Presiona `s` para guardar screenshots, `q` para salir

---

## 3. Parte 1B — Benchmark YOLO (YOLOv12 y YOLOv26)

Necesitas un video `.mp4`. Cópialo a esta carpeta (ej: `video.mp4`).

```bash
python3 practica4_1B_yolo_benchmark.py --video video.mp4
```

El script hará automáticamente:
1. YOLOv12 en CPU → guarda `out_yolo12n.pt_cpu.mp4`
2. YOLOv26 en CPU → guarda `out_yolo26n.pt_cpu.mp4`
3. YOLOv12 en GPU → guarda `out_yolo12n.pt_gpu.mp4`
4. YOLOv26 en GPU → guarda `out_yolo26n.pt_gpu.mp4`

Cada video muestra en overlay: **Modelo, Device, FPS, RAM, VRAM, MAC Address**.

---

## 4. Parte 1B — Benchmark Super Resolución (Real-ESRGAN)

Idealmente usa un video de **baja resolución (360p)** para que la VRAM no colapse con el upscale x4.

```bash
python3 practica4_1B_superres_benchmark.py --video video.mp4
```

Generará: `out_sr_cpu.mp4` y `out_sr_gpu.mp4`.

---

## 5. Parte 1C — OpenCV C++ Pipeline CPU vs GPU vs Híbrido

1. Entra a la carpeta C++:
   ```bash
   cd parte1c_cpp
   ```

2. Compila (solo una vez):
   ```bash
   ./build.sh
   ```

3. Ejecuta el benchmark:
   ```bash
   cd build
   ./benchmark_cv ../../video.mp4
   ```

   Verás en la consola un cuadro comparativo con tiempos por frame para los **3 pipelines**:
   - **CPU puro**
   - **GPU-only** (eficiente)
   - **CPU↔GPU híbrido** (ineficiente)

   Al final se abren 4 ventanas con los resultados visuales (toma captura de pantalla).
   También imprime la **reflexión** sobre cuándo vale la pena usar GPU.

---

## 6. Checklist de Evidencias para la Grabación

Mientras grabas la pantalla (OBS Studio), asegúrate de que se vea:

- [x] Terminal con `watch -n 1 nvidia-smi` visible (uso de VRAM)
- [x] FPS en GPU vs CPU (overlay en cada script)
- [x] Uso de memoria RAM (overlay en cada script)
- [x] MAC Address del computador (overlay en cada script)
- [x] Ventanas de resultado visual de la Parte 1C (4 ventanas)
- [x] Resumen de consola con el Speedup de la Parte 1C
