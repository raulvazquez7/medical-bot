"""
Ingestion service orchestrating the complete RAG pipeline.
Coordinates PDF parsing, chunking, embedding generation, and database ingestion.
"""

import logging
from pathlib import Path
from supabase import Client
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from src.services.pdf_service import PDFService, PDFParsingError
from src.services.chunking_service import ChunkingService


class IngestionError(Exception):
    """Raised when ingestion pipeline fails."""


class IngestionService:
    """
    Orchestrates the complete ingestion pipeline for medicine leaflets.
    Handles PDF → Markdown → Chunks → Embeddings → Database.
    """

    def __init__(
        self,
        pdf_service: PDFService,
        chunking_service: ChunkingService,
        embeddings_model: Embeddings,
        supabase_client: Client,
    ):
        """
        Initialize ingestion service.

        Args:
            pdf_service: Service for PDF parsing
            chunking_service: Service for chunking
            embeddings_model: Embeddings model instance
            supabase_client: Supabase client for database operations
        """
        self.pdf_service = pdf_service
        self.chunking_service = chunking_service
        self.embeddings_model = embeddings_model
        self.supabase = supabase_client
        logging.info("Ingestion service initialized")

    def run_pipeline(
        self,
        pdf_path: str,
        markdown_path: str,
        force_reparse: bool = False,
    ) -> dict:
        """
        Executes complete ingestion pipeline for a single PDF.

        Args:
            pdf_path: Path to input PDF file
            markdown_path: Path to save/load markdown
            force_reparse: If True, re-parse PDF even if markdown exists

        Returns:
            Dictionary with pipeline statistics

        Raises:
            IngestionError: If pipeline fails
        """
        pdf_filename = Path(pdf_path).name
        md_filename = Path(markdown_path).name

        logging.info(f"Starting pipeline for: {pdf_filename}")

        # Step 1: PDF to Markdown (with cache)
        markdown_text = self._parse_or_load_markdown(
            pdf_path, markdown_path, force_reparse
        )

        # Step 2: Extract and standardize medicine name
        medicine_name = self._extract_medicine_name(markdown_text, pdf_filename)

        # Step 3: Create chunks
        chunks = self._create_chunks(markdown_text, md_filename, medicine_name)

        # Step 4: Generate embeddings
        embeddings_list = self._generate_embeddings(chunks, md_filename)

        # Step 5: Clean old data
        self._cleanup_old_data(md_filename)

        # Step 6: Ingest new data
        self._ingest_to_database(chunks, embeddings_list, pdf_filename)

        stats = {
            "pdf_file": pdf_filename,
            "medicine_name": medicine_name,
            "total_chunks": len(chunks),
            "status": "success",
        }

        logging.info(f"Pipeline completed successfully for {pdf_filename}")
        return stats

    def _parse_or_load_markdown(
        self, pdf_path: str, markdown_path: str, force_reparse: bool
    ) -> str:
        """
        Parses PDF or loads cached markdown.

        Args:
            pdf_path: Path to PDF
            markdown_path: Path to markdown
            force_reparse: Force re-parsing

        Returns:
            Markdown content

        Raises:
            IngestionError: If parsing fails
        """
        markdown_path_obj = Path(markdown_path)

        if not force_reparse and markdown_path_obj.exists():
            logging.info(f"Markdown cache found: {markdown_path}")
            with open(markdown_path, "r", encoding="utf-8") as f:
                return f.read()

        logging.info("Parsing PDF to Markdown...")
        markdown_path_obj.parent.mkdir(parents=True, exist_ok=True)

        try:
            markdown_text = self.pdf_service.parse_pdf_to_markdown(
                pdf_path, str(markdown_path)
            )
            return markdown_text
        except PDFParsingError as e:
            raise IngestionError(f"PDF parsing failed: {e}") from e

    def _extract_medicine_name(
        self, markdown_text: str, pdf_filename: str
    ) -> str:
        """
        Extracts and standardizes medicine name.

        Args:
            markdown_text: Markdown content
            pdf_filename: PDF filename for fallback

        Returns:
            Standardized medicine name
        """
        raw_name = self.chunking_service.extract_medicine_name(
            markdown_text, pdf_filename
        )
        medicine_name = self.chunking_service.standardize_medicine_name(raw_name)
        logging.info(f"Medicine name standardized to: '{medicine_name}'")
        return medicine_name

    def _create_chunks(
        self, markdown_text: str, md_filename: str, medicine_name: str
    ) -> list[Document]:
        """
        Creates semantic chunks from markdown.

        Args:
            markdown_text: Markdown content
            md_filename: Markdown filename for metadata
            medicine_name: Standardized medicine name

        Returns:
            List of document chunks

        Raises:
            IngestionError: If no chunks created
        """
        logging.info("Creating semantic chunks...")
        blocks = self.chunking_service.markdown_to_semantic_blocks(markdown_text)

        chunks = self.chunking_service.create_sentence_window_chunks(
            blocks=blocks, source_file=md_filename, medicine_name=medicine_name
        )

        # Validate chunks
        if not chunks:
            raise IngestionError("No chunks generated from document")
        
        if len(chunks) > 10000:
            logging.warning(
                f"Very large number of chunks: {len(chunks)}. "
                "Consider adjusting chunk size."
            )
        
        # Validate chunk content
        for i, chunk in enumerate(chunks):
            if not chunk.page_content or not chunk.page_content.strip():
                raise IngestionError(f"Empty chunk at index {i}")
            if "medicine_name" not in chunk.metadata:
                raise IngestionError(f"Chunk {i} missing medicine_name in metadata")

        logging.info(f"Created {len(chunks)} chunks")
        return chunks

    def _generate_embeddings(
        self, chunks: list[Document], md_filename: str
    ) -> list[list[float]]:
        """
        Generates embeddings for chunks.

        Args:
            chunks: Document chunks
            md_filename: Markdown filename for error messages

        Returns:
            List of embedding vectors

        Raises:
            IngestionError: If embedding generation fails
        """
        logging.info("Generating embeddings...")
        texts_to_embed = [doc.page_content for doc in chunks]

        try:
            embeddings_list = self.embeddings_model.embed_documents(texts_to_embed)
            
            # Validate embeddings
            if len(embeddings_list) != len(chunks):
                raise IngestionError(
                    f"Embeddings/chunks mismatch: got {len(embeddings_list)} embeddings "
                    f"for {len(chunks)} chunks"
                )
            
            # Validate embedding dimensions (if we got any)
            if embeddings_list and not embeddings_list[0]:
                raise IngestionError("Generated empty embeddings")
            
            logging.info(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list
        except IngestionError:
            raise  # Re-raise validation errors
        except Exception as e:
            logging.error(f"Embedding generation failed: {e}", exc_info=True)
            raise IngestionError(
                f"Embedding generation failed for {md_filename}: {e}"
            ) from e

    def _cleanup_old_data(self, md_filename: str) -> None:
        """
        Removes old records from database for the same source file.

        Args:
            md_filename: Source filename to clean up
        """
        logging.info(f"Cleaning old records for '{md_filename}'...")
        try:
            self.supabase.table("documents").delete().eq(
                "metadata->>source", md_filename
            ).execute()
            logging.info("Old records cleaned successfully")
        except Exception as e:
            logging.warning(f"Cleanup warning (non-critical): {e}")

    def _ingest_to_database(
        self,
        chunks: list[Document],
        embeddings_list: list[list[float]],
        pdf_filename: str,
    ) -> None:
        """
        Ingests chunks and embeddings to Supabase.

        Args:
            chunks: Document chunks
            embeddings_list: Embedding vectors
            pdf_filename: PDF filename for error messages

        Raises:
            IngestionError: If database insertion fails
        """
        logging.info(f"Ingesting {len(chunks)} records to database...")

        records_to_insert = [
            {
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embeddings_list[i],
            }
            for i, chunk in enumerate(chunks)
        ]

        try:
            self.supabase.table("documents").insert(records_to_insert).execute()
            logging.info(
                f"Successfully ingested {len(records_to_insert)} records "
                f"for {pdf_filename}"
            )
        except Exception as e:
            logging.error(f"Database insertion failed: {e}", exc_info=True)
            raise IngestionError(
                f"Failed to insert data for {pdf_filename}: {e}"
            ) from e
