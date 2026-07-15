/*
 * ===========================================================================
 * Práctica 4 – Parte 1C: Preprocesamiento de Imágenes CPU vs GPU (CUDA)
 * Asignatura : Visión por Computador
 * Docente    : Ing. Vladimir Robles Bykbaev
 * 
 * Operaciones implementadas (según rúbrica):
 *   1. Suavizado              – Filtro Gaussiano
 *   2. Operaciones morfológicas – Erosión Y Dilatación
 *   3. Detección de bordes    – Canny
 *   4. Ecualización del histograma
 *
 * Se implementan TRES pipelines:
 *   A) CPU puro
 *   B) GPU-only (eficiente): un solo upload, todas las ops en GPU, un solo download
 *   C) CPU↔GPU híbrido (ineficiente): upload/download en CADA operación
 *
 * El pipeline C existe para responder la reflexión de la rúbrica:
 *   "¿Cuál es la diferencia entre pipeline CPU↔GPU y pipeline GPU-only?
 *    ¿Cuándo vale la pena usar la GPU?"
 *
 * Compilar:  mkdir build && cd build && cmake .. && make
 * Ejecutar:  ./benchmark_cv <ruta_video_o_imagen>
 * ===========================================================================
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

// ──────────────────────────── Utilidad de medición ────────────────────────────
struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> t0;
    void tic() { t0 = std::chrono::high_resolution_clock::now(); }
    float toc() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(t1 - t0).count();
    }
};

// ──────────────────── Pipeline A: CPU puro ────────────────────────────────────
void pipelineCPU(const Mat& input, Mat& output) {
    Mat gray, equalized, blur, eroded, dilated, edges;

    // 1. Escala de grises
    cvtColor(input, gray, COLOR_BGR2GRAY);

    // 2. Ecualización del histograma
    equalizeHist(gray, equalized);

    // 3. Suavizado (Filtro Gaussiano)
    GaussianBlur(equalized, blur, Size(5, 5), 1.5);

    // 4a. Operación morfológica: Erosión
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(blur, eroded, element);

    // 4b. Operación morfológica: Dilatación
    dilate(eroded, dilated, element);

    // 5. Detección de bordes (Canny)
    Canny(dilated, edges, 50, 150);

    output = edges;
}

// ──────────── Pipeline B: GPU-only (eficiente, como pide la rúbrica) ─────────
void pipelineGPU_only(const Mat& frame_cpu, Mat& output_cpu) {
    cuda::GpuMat d_frame, d_gray, d_equalized, d_blur, d_eroded, d_dilated, d_edges;

    // === UN SOLO UPLOAD al inicio ===
    d_frame.upload(frame_cpu);

    // 1. Escala de grises                                    [GPU]
    cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY);

    // 2. Ecualización del histograma                         [GPU]
    cuda::equalizeHist(d_gray, d_equalized);

    // 3. Suavizado (Filtro Gaussiano)                        [GPU]
    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(
        d_equalized.type(), d_equalized.type(), Size(5,5), 1.5);
    gauss->apply(d_equalized, d_blur);

    // 4a. Operación morfológica: Erosión                     [GPU]
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Ptr<cuda::Filter> erodeFilter = cuda::createMorphologyFilter(
        MORPH_ERODE, d_blur.type(), element);
    erodeFilter->apply(d_blur, d_eroded);

    // 4b. Operación morfológica: Dilatación                  [GPU]
    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(
        MORPH_DILATE, d_eroded.type(), element);
    dilateFilter->apply(d_eroded, d_dilated);

    // 5. Detección de bordes (Canny)                         [GPU]
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_dilated, d_edges);

    // === UN SOLO DOWNLOAD al final ===
    d_edges.download(output_cpu);
}

// ──── Pipeline C: CPU↔GPU híbrido (INEFICIENTE – para demostrar la diff) ────
void pipelineHybrid(const Mat& frame_cpu, Mat& output_cpu) {
    cuda::GpuMat d_temp_in, d_temp_out;
    Mat temp;

    // 1. Escala de grises — subir, procesar, bajar
    d_temp_in.upload(frame_cpu);
    cuda::cvtColor(d_temp_in, d_temp_out, COLOR_BGR2GRAY);
    d_temp_out.download(temp);                              // ← download innecesario

    // 2. Ecualización del histograma — subir, procesar, bajar
    d_temp_in.upload(temp);                                 // ← upload innecesario
    cuda::equalizeHist(d_temp_in, d_temp_out);
    d_temp_out.download(temp);                              // ← download innecesario

    // 3. Suavizado (Filtro Gaussiano) — subir, procesar, bajar
    d_temp_in.upload(temp);                                 // ← upload innecesario
    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(
        d_temp_in.type(), d_temp_in.type(), Size(5,5), 1.5);
    gauss->apply(d_temp_in, d_temp_out);
    d_temp_out.download(temp);                              // ← download innecesario

    // 4a. Erosión — subir, procesar, bajar
    d_temp_in.upload(temp);                                 // ← upload innecesario
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Ptr<cuda::Filter> erodeF = cuda::createMorphologyFilter(
        MORPH_ERODE, d_temp_in.type(), element);
    erodeF->apply(d_temp_in, d_temp_out);
    d_temp_out.download(temp);                              // ← download innecesario

    // 4b. Dilatación — subir, procesar, bajar
    d_temp_in.upload(temp);                                 // ← upload innecesario
    Ptr<cuda::Filter> dilateF = cuda::createMorphologyFilter(
        MORPH_DILATE, d_temp_in.type(), element);
    dilateF->apply(d_temp_in, d_temp_out);
    d_temp_out.download(temp);                              // ← download innecesario

    // 5. Canny — subir, procesar, bajar
    d_temp_in.upload(temp);                                 // ← upload innecesario
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_temp_in, d_temp_out);
    d_temp_out.download(output_cpu);                        // download final
}

// ─────────────────────────────── MAIN ────────────────────────────────────────
int main(int argc, char** argv) {

    cout << "============================================================" << endl;
    cout << "  Practica 4 - Parte 1C: OpenCV C++ Preprocesamiento" << endl;
    cout << "  Pipeline CPU  vs  GPU-only  vs  CPU<->GPU hibrido" << endl;
    cout << "============================================================" << endl;

    // --- Verificar CUDA ---
    int num_devices = cuda::getCudaEnabledDeviceCount();
    if (num_devices <= 0) {
        cerr << "[ERROR] No se detectaron dispositivos CUDA." << endl;
        cerr << "        Ejecutar en el laboratorio con GPUs NVIDIA." << endl;
        return -1;
    }
    cuda::DeviceInfo info(cuda::getDevice());
    cout << "GPU detectada: " << info.name() << endl;
    cout << "VRAM total:    " << info.totalMemory() / (1024*1024) << " MB" << endl;

    if (argc < 2) {
        cout << "Uso: ./benchmark_cv <ruta_al_video_o_imagen>" << endl;
        return -1;
    }

    // --- Abrir fuente de video o imagen ---
    String path = argv[1];
    VideoCapture cap(path);
    Mat static_img;
    bool is_video = true;

    if (!cap.isOpened()) {
        static_img = imread(path);
        if (static_img.empty()) {
            cerr << "Error: No se pudo cargar " << path << endl;
            return -1;
        }
        is_video = false;
        cout << "Modo: Imagen Estatica (" << static_img.cols << "x" << static_img.rows << ")" << endl;
    } else {
        int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
        int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
        cout << "Modo: Video (" << w << "x" << h << ")" << endl;
    }

    Mat frame, out_cpu, out_gpu, out_hybrid;
    Timer timer;

    vector<float> times_cpu, times_gpu, times_hybrid;

    // --- Warm-up de CUDA (la primera llamada siempre es lenta) ---
    cout << "Warm-up de CUDA..." << endl;
    if (is_video) cap.read(frame); else frame = static_img.clone();
    if (!frame.empty()) {
        try {
            pipelineGPU_only(frame, out_gpu);
            pipelineHybrid(frame, out_hybrid);
        } catch(const cv::Exception& e) {
            cerr << "Error CUDA: " << e.what() << endl;
            return -1;
        }
    }

    // --- Benchmark ---
    int frames_limit = 100;
    int count = 0;

    cout << "\nIniciando benchmark (" << frames_limit << " frames max)..." << endl;
    cout << "-----------------------------------------------------------" << endl;
    cout << setw(6) << "Frame"
         << setw(12) << "CPU (ms)"
         << setw(16) << "GPU-only (ms)"
         << setw(16) << "Hybrid (ms)" << endl;
    cout << "-----------------------------------------------------------" << endl;

    while (count < frames_limit) {
        if (is_video) {
            cap >> frame;
            if (frame.empty()) break;
        } else {
            frame = static_img.clone();
        }

        // A) CPU puro
        timer.tic();
        pipelineCPU(frame, out_cpu);
        float t_cpu = timer.toc();
        times_cpu.push_back(t_cpu);

        // B) GPU-only (eficiente)
        timer.tic();
        pipelineGPU_only(frame, out_gpu);
        float t_gpu = timer.toc();
        times_gpu.push_back(t_gpu);

        // C) Híbrido CPU↔GPU (ineficiente)
        timer.tic();
        pipelineHybrid(frame, out_hybrid);
        float t_hyb = timer.toc();
        times_hybrid.push_back(t_hyb);

        count++;

        cout << setw(6) << count
             << setw(12) << fixed << setprecision(2) << t_cpu
             << setw(16) << t_gpu
             << setw(16) << t_hyb << endl;
    }

    // --- Calcular promedios ---
    float avg_cpu = 0, avg_gpu = 0, avg_hyb = 0;
    for (int i = 0; i < count; i++) {
        avg_cpu += times_cpu[i];
        avg_gpu += times_gpu[i];
        avg_hyb += times_hybrid[i];
    }
    avg_cpu /= count;
    avg_gpu /= count;
    avg_hyb /= count;

    cout << "\n============================================================" << endl;
    cout << "              RESUMEN DE RESULTADOS" << endl;
    cout << "============================================================" << endl;
    cout << "Frames procesados    : " << count << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << "Pipeline CPU         : " << fixed << setprecision(2) << avg_cpu
         << " ms/frame  (" << setprecision(1) << 1000.0f/avg_cpu << " FPS)" << endl;
    cout << "Pipeline GPU-only    : " << setprecision(2) << avg_gpu
         << " ms/frame  (" << setprecision(1) << 1000.0f/avg_gpu << " FPS)" << endl;
    cout << "Pipeline CPU<->GPU   : " << setprecision(2) << avg_hyb
         << " ms/frame  (" << setprecision(1) << 1000.0f/avg_hyb << " FPS)" << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << "Speedup GPU-only/CPU : " << setprecision(2) << avg_cpu / avg_gpu << "x" << endl;
    cout << "Speedup Hybrid/CPU   : " << setprecision(2) << avg_cpu / avg_hyb << "x" << endl;
    cout << "Overhead hibrido vs GPU-only: " << setprecision(2) << avg_hyb / avg_gpu << "x mas lento" << endl;
    cout << "============================================================" << endl;

    cout << "\n--- REFLEXION (para el informe) ---" << endl;
    cout << "El pipeline GPU-only es " << setprecision(2) << avg_hyb / avg_gpu
         << "x mas rapido que el hibrido CPU<->GPU." << endl;
    cout << "Esto demuestra que las transferencias de datos entre CPU y GPU" << endl;
    cout << "(upload/download) son un cuello de botella significativo." << endl;
    cout << "Es mas eficiente mantener todos los datos en la GPU y solo" << endl;
    cout << "transferir al inicio (upload) y al final (download)." << endl;
    cout << "La GPU vale la pena cuando:" << endl;
    cout << "  - Se encadenan multiples operaciones sobre la misma imagen" << endl;
    cout << "  - Se procesan frames de video en tiempo real" << endl;
    cout << "  - La resolucion de la imagen es grande" << endl;

    // --- Mostrar resultados visuales (capturas de pantalla) ---
    cout << "\nMostrando ventanas de resultados. Presiona cualquier tecla para cerrar." << endl;

    // Redimensionar para que quepan en pantalla
    Mat display_orig, display_cpu, display_gpu, display_hybrid;
    int disp_w = 480;
    float scale_factor = (float)disp_w / frame.cols;
    int disp_h = (int)(frame.rows * scale_factor);
    Size display_size(disp_w, disp_h);

    resize(frame, display_orig, display_size);
    resize(out_cpu, display_cpu, display_size);
    resize(out_gpu, display_gpu, display_size);
    resize(out_hybrid, display_hybrid, display_size);

    imshow("1. Original", display_orig);
    imshow("2. Pipeline CPU (Canny)", display_cpu);
    imshow("3. Pipeline GPU-only (Canny)", display_gpu);
    imshow("4. Pipeline CPU<->GPU Hibrido (Canny)", display_hybrid);

    waitKey(0);
    destroyAllWindows();

    // --- Guardar imágenes de resultado como evidencia ---
    imwrite("resultado_original.jpg", frame);
    imwrite("resultado_cpu.jpg", out_cpu);
    imwrite("resultado_gpu_only.jpg", out_gpu);
    imwrite("resultado_hibrido.jpg", out_hybrid);
    cout << "Imagenes guardadas: resultado_original.jpg, resultado_cpu.jpg," << endl;
    cout << "                    resultado_gpu_only.jpg, resultado_hibrido.jpg" << endl;

    return 0;
}
