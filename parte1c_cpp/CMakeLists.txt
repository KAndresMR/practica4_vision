cmake_minimum_required(VERSION 3.10)
project(Practica4_1C_CUDA)

set(CMAKE_CXX_STANDARD 14)

# Buscar OpenCV
find_package(OpenCV REQUIRED)

# Incluir los headers de OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Crear el ejecutable
add_executable(benchmark_cv main.cpp)

# Enlazar con las librerías de OpenCV
target_link_libraries(benchmark_cv ${OpenCV_LIBS})

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
