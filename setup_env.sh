#!/bin/bash
set -e  # Detiene la ejecución si ocurre algún error

ENV_NAME="vision_p4"
PYTHON_VERSION="3.10"

echo "========================================================="
echo "  Configuración del Entorno Virtual - Práctica 4"
echo "========================================================="

# 1. Comprobar si conda está instalado
if ! command -v conda &> /dev/null
then
    echo "❌ Error: Conda no está instalado o no está en el PATH."
    echo "Por favor instala Miniconda o Anaconda antes de ejecutar este script."
    exit 1
fi
echo "✅ Conda detectado."

# 2. Aceptar Términos de Servicio de Anaconda (Previene error de CondaToS)
echo "📜 Aceptando Términos de Servicio (si es requerido)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 3. Crear el entorno virtual
echo "📦 Creando el entorno virtual '$ENV_NAME' con Python $PYTHON_VERSION..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# 4. Inicializar conda en este script
eval "$(conda shell.bash hook)"

# 5. Activar el entorno
echo "🔄 Activando el entorno '$ENV_NAME'..."
conda activate $ENV_NAME

# 6. Instalar dependencias
echo "📥 Instalando librerías desde requirements_lab.txt..."
python -m pip install --upgrade pip
# Workaround para evitar el error de __version__ con basicsr en instalaciones limpias
python -m pip install tb-nightly || true
python -m pip install -r requirements_lab.txt

echo "========================================================="
echo "🎉 ¡Entorno configurado con éxito!"
echo "Para comenzar a usar el proyecto, ejecuta en tu terminal:"
echo ""
echo "    conda activate $ENV_NAME"
echo ""
echo "Y luego puedes seguir el paso a paso del README_LAB.md."
echo "========================================================="
