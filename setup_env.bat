@echo off
chcp 65001 >nul
echo =========================================================
echo   Configuracion del Entorno Virtual - Practica 4
echo =========================================================

:: 1. Comprobar si conda está instalado
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda no esta instalado o no esta en el PATH de Windows.
    echo Por favor instala Miniconda o Anaconda antes de ejecutar este script.
    pause
    exit /b 1
)
echo [OK] Conda detectado.

:: 2. Crear el entorno virtual
set ENV_NAME=vision_p4
set PYTHON_VERSION=3.10
echo [INFO] Creando el entorno virtual '%ENV_NAME%' con Python %PYTHON_VERSION%...
call conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%

:: 3. Activar el entorno
echo [INFO] Activando el entorno...
call conda activate %ENV_NAME%

:: 4. Instalar dependencias
echo [INFO] Instalando librerias desde requirements_lab.txt...
python -m pip install --upgrade pip
pip install -r requirements_lab.txt

echo =========================================================
echo  Entorno configurado con exito!
echo  Para comenzar a usar el proyecto, abre tu Anaconda Prompt y ejecuta:
echo.
echo     conda activate %ENV_NAME%
echo.
echo  Y luego puedes seguir el paso a paso del README_LAB.md.
echo =========================================================
pause
