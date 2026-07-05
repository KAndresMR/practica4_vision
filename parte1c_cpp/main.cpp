#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

// Variables para medir tiempo
struct Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    std::chrono::duration<float> duration;

    void tic() {
        start = std::chrono::high_resolution_clock::now();
    }

    float toc() {
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        return duration.count() * 1000.0f; // retorna en milisegundos
    }
};

void processPipelineCPU(const Mat& input, Mat& output) {
    Mat gray, blur, equalized, morph, edges;
    
    // 1. Escala de grises
    cvtColor(input, gray, COLOR_BGR2GRAY);
    
    // 2. Ecualización del histograma
    equalizeHist(gray, equalized);
    
    // 3. Suavizado (Filtro Gaussiano)
    GaussianBlur(equalized, blur, Size(5, 5), 1.5);
    
    // 4. Operaciones morfológicas (Dilatación)
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(blur, morph, element);
    
    // 5. Detección de bordes (Canny)
    Canny(morph, edges, 50, 150);
    
    output = edges;
}

void processPipelineGPU(const Mat& frame_cpu, Mat& output_cpu) {
    cuda::GpuMat d_frame, d_gray, d_equalized, d_blur, d_morph, d_edges;
    
    // --- INICIO PIPELINE GPU-ONLY ---
    
    // Subir a la GPU
    d_frame.upload(frame_cpu);
    
    // 1. Escala de grises
    cuda::cvtColor(d_frame, d_gray, COLOR_BGR2GRAY);
    
    // 2. Ecualización del histograma
    cuda::equalizeHist(d_gray, d_equalized);
    
    // 3. Suavizado (Filtro Gaussiano)
    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(d_equalized.type(), d_equalized.type(), Size(5,5), 1.5);
    gauss->apply(d_equalized, d_blur);
    
    // 4. Operaciones morfológicas (Dilatación)
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, d_blur.type(), element);
    dilateFilter->apply(d_blur, d_morph);
    
    // 5. Detección de bordes (Canny)
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(50, 150);
    canny->detect(d_morph, d_edges);
    
    // Bajar a la CPU (SOLO AL FINAL)
    d_edges.download(output_cpu);
    
    // --- FIN PIPELINE GPU-ONLY ---
}

int main(int argc, char** argv) {
    cout << "=======================================" << endl;
    cout << "Practica 4 - Parte 1C: C++ OpenCV CUDA" << endl;
    cout << "=======================================" << endl;

    // Verificar si CUDA está disponible
    int num_devices = cuda::getCudaEnabledDeviceCount();
    if (num_devices <= 0) {
        cerr << "ADVERTENCIA: No se detectaron dispositivos CUDA o OpenCV no fue compilado con soporte CUDA." << endl;
        cerr << "Esta prueba fallará a menos que se ejecute en el laboratorio con GPUs NVIDIA." << endl;
    } else {
        cuda::DeviceInfo info(cuda::getDevice());
        cout << "Dispositivo CUDA detectado: " << info.name() << endl;
    }

    if (argc < 2) {
        cout << "Uso: ./benchmark_cv <ruta_al_video_o_imagen>" << endl;
        return -1;
    }

    String path = argv[1];
    VideoCapture cap(path);
    
    // Si no puede abrir como video, intentar cargar como imagen estática
    Mat static_img;
    bool is_video = true;
    
    if (!cap.isOpened()) {
        static_img = imread(path);
        if (static_img.empty()) {
            cerr << "Error: No se pudo cargar el archivo " << path << endl;
            return -1;
        }
        is_video = false;
        cout << "Modo: Imagen Estatica" << endl;
    } else {
        cout << "Modo: Video Streaming" << endl;
    }

    Mat frame, out_cpu, out_gpu;
    Timer timer;
    
    vector<float> cpu_times;
    vector<float> gpu_times;

    // Ejecutar warmup en GPU (la primera inicialización de CUDA siempre es muy lenta)
    cout << "Realizando Warmup de CUDA..." << endl;
    if (is_video) { cap.read(frame); } else { frame = static_img; }
    
    if (!frame.empty()) {
        try {
            processPipelineGPU(frame, out_gpu);
        } catch(const cv::Exception& e) {
            cerr << "Error de OpenCV/CUDA. Probablemente falta el soporte CUDA. Detalle: " << e.what() << endl;
            return -1;
        }
    }

    int frames_limit = 100; // Limitar para que el benchmark no sea eterno
    int count = 0;
    
    cout << "Iniciando benchmark CPU vs GPU..." << endl;
    
    while (true) {
        if (is_video) {
            cap >> frame;
            if (frame.empty() || count >= frames_limit) break;
        } else {
            frame = static_img;
            if (count >= frames_limit) break;
        }
        
        // Benchmark CPU
        timer.tic();
        processPipelineCPU(frame, out_cpu);
        float time_cpu = timer.toc();
        cpu_times.push_back(time_cpu);
        
        // Benchmark GPU
        timer.tic();
        processPipelineGPU(frame, out_gpu);
        float time_gpu = timer.toc();
        gpu_times.push_back(time_gpu);
        
        count++;
        
        // Mostrar resultados en consola
        cout << "Frame " << count << " | CPU: " << time_cpu << " ms | GPU: " << time_gpu << " ms" << endl;
        
        // Para visualización (solo mostramos el último frame si es imagen, si es video va rápido)
        // imshow("Original", frame);
        // imshow("Pipeline CPU", out_cpu);
        // imshow("Pipeline GPU-only", out_gpu);
        // if (waitKey(1) == 'q') break;
    }

    // Calcular promedios
    float avg_cpu = 0, avg_gpu = 0;
    for (int i = 0; i < count; i++) {
        avg_cpu += cpu_times[i];
        avg_gpu += gpu_times[i];
    }
    avg_cpu /= count;
    avg_gpu /= count;
    
    cout << "\n=======================================" << endl;
    cout << "         RESUMEN DE RESULTADOS         " << endl;
    cout << "=======================================" << endl;
    cout << "Frames procesados : " << count << endl;
    cout << "Tiempo Medio CPU  : " << avg_cpu << " ms/frame  (" << 1000.0f/avg_cpu << " FPS)" << endl;
    cout << "Tiempo Medio GPU  : " << avg_gpu << " ms/frame  (" << 1000.0f/avg_gpu << " FPS)" << endl;
    cout << "Speedup GPU/CPU   : " << avg_cpu / avg_gpu << "x mas rapido" << endl;
    
    // Guardar las imágenes resultantes
    imwrite("resultado_cpu.jpg", out_cpu);
    imwrite("resultado_gpu.jpg", out_gpu);
    cout << "Imagenes de muestra guardadas como resultado_cpu.jpg y resultado_gpu.jpg" << endl;

    return 0;
}
