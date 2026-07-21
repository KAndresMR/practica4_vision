# Práctica 4 - Evaluación de Modelos de IA en GPU vs CPU

Este repositorio contiene el código fuente y los scripts necesarios para ejecutar y evaluar las 3 partes de la Práctica 4 de Visión por Computador. Las pruebas están diseñadas para ejecutarse y ser probadas en equipos con aceleración gráfica (NVIDIA CUDA).

---

## 1. Instalación Rápida del Entorno (Miniconda)

Para evitar errores de versiones o dependencias al clonar este repositorio en otra máquina, preparé scripts automáticos que instalan todo lo necesario dentro de un entorno virtual de Conda.

Dependiendo del sistema operativo de la máquina donde se vaya a calificar/probar, ejecuta:

**Si usas Windows:**
Dale doble clic al archivo `setup_env.bat` o ejecútalo desde el Anaconda Prompt:
```cmd
setup_env.bat
```

**Si usas Linux (Ubuntu/Kubuntu) o macOS:**
Abre una terminal en la raíz de este proyecto y ejecuta:
```bash
bash setup_env.sh
```

Una vez que el script termine de descargar todo, **activa el entorno** antes de correr cualquier prueba:
```bash
conda activate vision_p4
```

---

## 2. Preparación de Evidencias (nvidia-smi)

Como parte de la rúbrica, es necesario demostrar el consumo de VRAM en tiempo real. Por ello, abre una **segunda terminal**, activa el entorno, y déjala a un lado de la pantalla ejecutando:

```bash
watch -n 1 nvidia-smi
```
*(Nota: En Windows puedes usar simplemente `nvidia-smi -l 1`)*

---

## 3. Ejecución Parte 1A: Segmentación con Webcam

Para probar el modelo de Transfer Learning (`best.pt`) que entrené específicamente para segmentar manzanas, bananas y naranjas:

**Prueba en GPU:**
```bash
python practica4_1A_export/webcam_fruit_seg.py --device gpu --model practica4_1A_export/best.pt
```

**Prueba en CPU:**
```bash
python practica4_1A_export/webcam_fruit_seg.py --device cpu --model practica4_1A_export/best.pt
```
*Puedes presionar la tecla `s` mientras corre el programa para guardar automáticamente una captura de pantalla de la detección.*

---

## 4. Ejecución Parte 1B: Benchmarks (YOLO y Super Resolución)

Asegúrate de tener el archivo `video_prueba.mp4` en la raíz de la carpeta. 

Para ejecutar las métricas de rendimiento de YOLO:
```bash
python practica4_1B_yolo_benchmark.py --video video_prueba.mp4
```
*(Este script procesará el video con YOLOv12 y YOLOv26, alternando entre CPU y GPU de forma automática. Al final generará los videos de salida con los FPS y la MAC Address impresos).*

Para ejecutar el modelo de Super Resolución:
```bash
python practica4_1B_superres_benchmark.py --video video_prueba.mp4
```

---

## 5. Ejecución Parte 1C: Pipeline C++ (OpenCV + CUDA)

Esta parte del proyecto evalúa los cuellos de botella al transferir datos. Aplica una secuencia de filtros (Gaussiano -> Morfología -> Canny -> Ecualización) comparando una arquitectura CPU pura, una Híbrida y una 100% en GPU.

1. Entra a la carpeta de C++ y compila el binario:
```bash
cd parte1c_cpp
./build.sh
cd build
```

2. Ejecuta el programa pasándole el video de prueba:
```bash
./benchmark_cv ../../video_prueba.mp4
```
Al finalizar, la consola imprimirá una tabla comparativa con los milisegundos de latencia por frame y una reflexión técnica sobre el cuello de botella del PCI-Express. También se abrirán las 4 ventanas visuales comprobando que los resultados de Canny son matemáticamente idénticos en todas las arquitecturas.
