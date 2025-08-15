#!/bin/bash

# Script para ejecutar el pipeline de ingesta para una lista predefinida de medicamentos.
# Asegúrate de que tu entorno virtual (.venv) está activado antes de ejecutar este script.

# Salir inmediatamente si un comando falla
set -e

# Lista de archivos PDF a procesar
FILES_TO_PROCESS=(
    "lexatin_3.pdf"
    "ibuprofeno_cinfa_600.pdf"
    "sintrom_4.pdf"
    "espidifen_600.pdf"
    "nolotil_575.pdf"
)

# Bucle para procesar cada archivo
for pdf_file in "${FILES_TO_PROCESS[@]}"; do
    echo ""
    echo "=================================================="
    echo "===> PROCESANDO: $pdf_file"
    echo "=================================================="
    
    python scripts/_03_ingest.py "$pdf_file"
done

echo ""
echo "=================================================="
echo "✅ Proceso de ingesta por lotes completado."
echo "=================================================="
