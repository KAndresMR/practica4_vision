#!/bin/bash
echo "Configurando y compilando el proyecto C++ Parte 1C..."

# Crear carpeta de build si no existe
mkdir -p build
cd build

# Ejecutar CMake y compilar
cmake ..
make -j4

echo "=========================================================="
echo "Compilación exitosa."
echo "Para probar en el laboratorio ejecuta:"
echo "cd build"
echo "./benchmark_cv ../alguna_imagen.jpg   # o usar un video .mp4"
echo "=========================================================="
