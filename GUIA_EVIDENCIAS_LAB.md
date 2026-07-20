# 📝 Guía Oficial de Pruebas y Evidencias - Práctica 4

**Materia:** Visión por Computador  
**Proyecto:** Evaluación de Modelos de IA y Preprocesamiento en GPU vs CPU (CUDA)  

Este documento es una guía profesional paso a paso diseñada estrictamente para cumplir al 100% con los requerimientos de la rúbrica de la **Práctica 4**. Síguela al pie de la letra en el Laboratorio (Cómputo 8) para asegurar que tomas todas las evidencias, capturas y grabaciones necesarias para tu informe.

---

## 🎒 0. Requisitos Previos (Lo que debes llevar al Lab)
Antes de sentarte en la computadora del laboratorio, asegúrate de tener:
1. **Todo el código fuente** (esta carpeta `practica4/`).
2. **Tu modelo entrenado** `best.pt` (de la parte 1A) guardado dentro de la carpeta.
3. **Un video corto (.mp4)** de 5 a 10 segundos grabado por ti (autos en la calle, frutas, objetos de escritorio). Nómbralo `video_prueba.mp4` y ponlo en esta carpeta.
4. **Software de grabación:** Asegúrate de que la PC tenga OBS Studio o usa la herramienta de grabación de Windows/Linux para grabar toda tu pantalla.

### 🛠 Configuración Inicial (¡Graba desde aquí!)
Inicia la grabación de tu pantalla y abre **dos terminales**.
En la **Terminal 1**, instala las dependencias:
```bash
pip install -r requirements_lab.txt
```
En la **Terminal 2**, ejecuta el monitor de la GPU de Nvidia. **Mantenla visible durante toda tu grabación** (cumple el requisito literal de la rúbrica de mostrar `nvidia-smi`):
```bash
watch -n 1 nvidia-smi
```

---

## 🟢 PARTE 1A: Fine-tuning y Detección por Webcam

**🎯 Qué pide la rúbrica:** Emplear una red (YOLO) con transfer learning para segmentar objetos de un dataset propio a través de la webcam, comprobando el rendimiento GPU vs CPU.

### ¿Qué debes ejecutar?
Abre la **Terminal 1** y ejecuta la prueba con GPU:
```bash
python3 webcam_fruit_seg.py --device gpu --model best.pt
```
Pon una fruta (manzana, banana, naranja) frente a la cámara. Presiona `q` para cerrar cuando termines.
Luego, ejecuta la prueba con CPU:
```bash
python3 webcam_fruit_seg.py --device cpu --model best.pt
```

### 📸 Evidencias a recopilar para la Parte 1A:
- [ ] **Grabación en video:** Debe verse cómo en GPU la detección fluye rápido y en CPU se congela o va lento.
- [ ] **Captura de pantalla (Screenshot):** Mientras se ejecuta, presiona la tecla `s` para que el script guarde automáticamente una foto de la detección. Hazlo tanto para CPU como GPU.
- [ ] **Verificación de Rúbrica:** El overlay en pantalla mostrará tu MAC Address, FPS y RAM, validando que el trabajo es de autoría única.

---

## 🟡 PARTE 1B: Benchmarks YOLOv12/v26 y Super Resolución

**🎯 Qué pide la rúbrica:** Realizar pruebas de rendimiento (FPS, RAM, VRAM, MAC) comparando CPU vs GPU con dos redes de detección (YOLOv12 y YOLOv26) y una red de super resolución reciente (Real-ESRGAN o equivalente).

### ¿Qué debes ejecutar?
Asegúrate de tener tu `video_prueba.mp4` en la carpeta. Ejecuta el benchmark de YOLO:
```bash
python3 practica4_1B_yolo_benchmark.py --video video_prueba.mp4
```
*Nota: El script correrá 4 veces de forma automática (YOLOv12 en CPU, YOLOv26 en CPU, YOLOv12 en GPU, YOLOv26 en GPU).*

Luego, ejecuta el benchmark de Super Resolución:
```bash
python3 practica4_1B_superres_benchmark.py --video video_prueba.mp4
```

### 📸 Evidencias a recopilar para la Parte 1B:
- [ ] **Grabación en video:** La terminal con `nvidia-smi` debe verse de fondo mientras el script procesa.
- [ ] **Captura de pantalla:** Toma foto a la terminal cuando los scripts terminen. Imprimirán un **Resumen de Resultados** con los FPS promedio.
- [ ] **Videos generados:** Los scripts guardarán videos resultantes (ej. `out_yolo12n_gpu.mp4`, `out_sr_cpu.mp4`). En el overlay de estos videos están incrustados los FPS, RAM y la MAC Address. **Sube uno de estos clips como evidencia de que cumpliste los requerimientos visuales.**

---

## 🔵 PARTE 1C: Pipeline C++ OpenCV con CUDA

**🎯 Qué pide la rúbrica:** Código nativo en OpenCV C++ que haga Suavizado Gaussiano, Morfología (Erosión/Dilatación), Canny y Ecualización de histograma. Debe implementarse un "Pipeline GPU-only" y compararse cualitativa y cuantitativamente contra la CPU.

### ¿Qué debes ejecutar?
Primero, debes compilar el proyecto (solo se hace una vez):
```bash
cd parte1c_cpp
./build.sh
cd build
```
Ahora, ejecuta el binario pasando tu video o una imagen:
```bash
./benchmark_cv ../../video_prueba.mp4
```

### 📸 Evidencias a recopilar para la Parte 1C:
- [ ] **Captura de la Terminal:** Cuando el programa termine, imprimirá una tabla comparando los **ms por frame** de la CPU, del Pipeline GPU-only, y de un pipeline híbrido (CPU↔GPU). **Toma una captura obligatoria de esto** para incluirla en la sección de análisis cuantitativo del informe.
- [ ] **Captura de las Ventanas Visuales:** El programa abrirá 4 ventanas mostrando los bordes de Canny finales. **Toma una captura de pantalla de estas ventanas** para la comparación cualitativa.
- [ ] **Reflexión para el informe:** Copia el texto que la consola imprime al final titulado `"--- REFLEXION ---"`. Ahí están los datos para que redactes tus conclusiones sobre por qué el pipeline "GPU-only" evita cuellos de botella respecto a estar subiendo y bajando datos de la CPU (Híbrido).

---

## ✅ Checklist Final antes de apagar la computadora del Lab

Antes de irte, revisa que en tu pendrive tengas:
1. [ ] El video o videos grabados con OBS Studio.
2. [ ] Las capturas de pantalla de la webcam (imágenes de la 1A).
3. [ ] Los videos de resultado `.mp4` generados automáticamente por los scripts de la parte 1B.
4. [ ] La captura de pantalla de los tiempos (Speedup) y ventanas generadas por el programa de C++ (Parte 1C).
5. [ ] Asegurarte de que en los videos se observe claramente la terminal con el comando `nvidia-smi` y la MAC Address impresa.

**¡Con esto tienes asegurado el 100% de la nota en cumplimiento técnico de la rúbrica!** 🎉
