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

:: 2. Aceptar Términos de Servicio de Anaconda (Previene error de CondaToS)
echo [INFO] Aceptando Terminos de Servicio (si es requerido)...
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >nul 2>&1
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >nul 2>&1

:: 3. Crear el entorno virtual
set ENV_NAME=vision_p4
set PYTHON_VERSION=3.10
echo [INFO] Creando el entorno virtual '%ENV_NAME%' con Python %PYTHON_VERSION%...
call conda create -y -n %ENV_NAME% python=%PYTHON_VERSION%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo al crear el entorno Conda.
    pause
    exit /b 1
)

:: 4. Activar el entorno
echo [INFO] Activando el entorno...
call conda activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Fallo al activar el entorno Conda.
    pause
    exit /b 1
)

:: 5. Instalar dependencias
echo [INFO] Instalando librerias desde requirements_lab.txt...
python -m pip install --upgrade pip
python -m pip install tb-nightly
python -m pip install -r requirements_lab.txt

echo =========================================================
echo  Entorno configurado con exito!
echo  Para comenzar a usar el proyecto, abre tu Anaconda Prompt y ejecuta:
echo.
echo     conda activate %ENV_NAME%
echo.
echo  Y luego puedes seguir el paso a paso del README_LAB.md.
echo =========================================================
pause
