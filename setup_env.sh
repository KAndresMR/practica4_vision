#!/bin/bash
# Script automático para crear y configurar el entorno de la Práctica 4

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

# 2. Crear el entorno virtual
echo "📦 Creando el entorno virtual '$ENV_NAME' con Python $PYTHON_VERSION..."
# El flag -y acepta automáticamente, --force permite sobreescribir si ya existía a medias
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# 3. Inicializar conda en este script para poder activar entornos
eval "$(conda shell.bash hook)"

# 4. Activar el entorno
echo "🔄 Activando el entorno '$ENV_NAME'..."
conda activate $ENV_NAME

# 5. Instalar dependencias
echo "📥 Instalando librerías desde requirements_lab.txt..."
pip install --upgrade pip
pip install -r requirements_lab.txt

echo "========================================================="
echo "🎉 ¡Entorno configurado con éxito!"
echo "Para comenzar a usar el proyecto, ejecuta en tu terminal:"
echo ""
echo "    conda activate $ENV_NAME"
echo ""
echo "Y luego puedes seguir el paso a paso del README_LAB.md."
echo "========================================================="
