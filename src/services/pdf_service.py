"""
PDF service for converting medicine leaflets to structured Markdown.
Uses multimodal LLM to parse PDF images and extract hierarchical structure.
"""

import logging
import base64
import yaml
from pathlib import Path
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


def load_prompts() -> dict:
    """Loads prompts from YAML configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PROMPTS = load_prompts()


class ProspectusMarkdown(BaseModel):
    """
    Structured output schema for PDF parsing.
    The LLM will return the complete prospectus in Markdown format.
    """

    markdown_content: str = Field(
        description=(
            "Complete document content in Markdown format, "
            "following all formatting and structure rules"
        )
    )


class PDFParsingError(Exception):
    """Raised when PDF parsing fails."""


class PDFService:
    """
    Service for PDF document parsing using multimodal LLM.
    Converts medicine leaflet PDFs to structured Markdown.
    """

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize PDF service.

        Args:
            model_name: Multimodal model name (e.g., "gemini-1.5-flash")
            api_key: API key for the LLM provider
        """
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.structured_llm = self.llm.with_structured_output(ProspectusMarkdown)
        logging.info(f"PDF service initialized with model: {model_name}")

    def pdf_to_base64_images(self, pdf_path: str) -> list[str]:
        """
        Converts each page of a PDF to base64-encoded PNG images.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of base64-encoded image strings

        Raises:
            PDFParsingError: If PDF conversion fails
        """
        logging.info(f"Converting PDF to images: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            base64_images = []

            for page in doc:
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
                base64_images.append(base64_image)

            doc.close()
            logging.info(f"Converted {len(base64_images)} pages to images")
            return base64_images

        except Exception as e:
            logging.error(f"PDF conversion failed: {e}", exc_info=True)
            raise PDFParsingError(f"Failed to convert PDF: {e}") from e

    def parse_pdf_to_markdown(
        self, pdf_path: str, output_path: str
    ) -> str:
        """
        Main pipeline: converts PDF to markdown using LLM vision.

        Args:
            pdf_path: Path to input PDF file
            output_path: Path to save markdown output

        Returns:
            Extracted markdown content

        Raises:
            PDFParsingError: If parsing fails
        """
        base64_images = self.pdf_to_base64_images(pdf_path)

        system_message = SystemMessage(
            content=PROMPTS["pdf_parsing"]["system_prompt"]
        )

        human_message_content = [
            {"type": "image_url", "image_url": f"data:image/png;base64,{img}"}
            for img in base64_images
        ]

        human_message = HumanMessage(content=human_message_content)

        logging.info(
            f"Sending {len(base64_images)} pages to LLM for parsing..."
        )
        try:
            prompt = [system_message, human_message]
            response = self.structured_llm.invoke(prompt)
            markdown_content = response.markdown_content

            logging.info("LLM returned structured markdown successfully")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            logging.info(f"Markdown saved to: {output_path}")
            return markdown_content

        except Exception as e:
            logging.error(f"LLM parsing failed: {e}", exc_info=True)
            raise PDFParsingError(f"LLM parsing failed: {e}") from e
