#!/bin/bash

# Batch ingestion script for medicine leaflets.
# Processes multiple PDFs through the RAG pipeline.

set -e

FILES_TO_PROCESS=(
    "lexatin_3.pdf"
    "ibuprofeno_cinfa_600.pdf"
    "sintrom_4.pdf"
    "espidifen_600.pdf"
    "nolotil_575.pdf"
)

for pdf_file in "${FILES_TO_PROCESS[@]}"; do
    echo ""
    echo "=================================================="
    echo "===> Processing: $pdf_file"
    echo "=================================================="

    python scripts/ingest.py "$pdf_file"
done

echo ""
echo "=================================================="
echo "✅ Batch ingestion completed successfully."
echo "=================================================="
