"""
CLI for medicine leaflet ingestion pipeline.
Thin wrapper around IngestionService for command-line usage.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from supabase import create_client
from src import config
from src.models.embeddings import get_embeddings_model
from src.services.pdf_service import PDFService
from src.services.chunking_service import ChunkingService
from src.services.ingestion_service import IngestionService, IngestionError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> int:
    """
    Main CLI entry point for ingestion pipeline.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Ingest medicine leaflet PDF into RAG database"
    )
    parser.add_argument(
        "pdf_filename",
        type=str,
        help="PDF filename (must be in data/ folder)",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Force PDF re-parsing even if markdown exists",
    )
    args = parser.parse_args()

    try:
        logging.info("Initializing ingestion pipeline...")
        config.check_env_vars()

        supabase = create_client(
            config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY
        )

        embeddings = get_embeddings_model(
            provider=config.EMBEDDINGS_PROVIDER.lower(),
            model=config.EMBEDDINGS_MODEL,
            api_key=(
                config.GOOGLE_API_KEY
                if config.EMBEDDINGS_PROVIDER.lower() == "google"
                else config.OPENAI_API_KEY
            ),
        )

        pdf_service = PDFService(
            model_name=config.PDF_PARSE_MODEL,
            api_key=config.GOOGLE_API_KEY,
        )

        chunking_service = ChunkingService(window_size=2)

        ingestion_service = IngestionService(
            pdf_service=pdf_service,
            chunking_service=chunking_service,
            embeddings_model=embeddings,
            supabase_client=supabase,
        )

        pdf_path = os.path.join(config.DATA_PATH, args.pdf_filename)
        if not os.path.exists(pdf_path):
            logging.error(f"PDF file not found: {pdf_path}")
            return 1

        model_slug = config.PDF_PARSE_MODEL.replace(".", "-")
        md_filename = f"parsed_by_{model_slug}_{args.pdf_filename.replace('.pdf', '.md')}"
        markdown_path = os.path.join(config.MARKDOWN_PATH, md_filename)

        stats = ingestion_service.run_pipeline(
            pdf_path=pdf_path,
            markdown_path=markdown_path,
            force_reparse=args.force_reparse,
        )

        logging.info("=" * 60)
        logging.info("Ingestion completed successfully!")
        logging.info(f"Medicine: {stats['medicine_name']}")
        logging.info(f"Chunks created: {stats['total_chunks']}")
        logging.info("=" * 60)

        return 0

    except IngestionError as e:
        logging.error(f"Ingestion failed: {e}")
        return 1
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
