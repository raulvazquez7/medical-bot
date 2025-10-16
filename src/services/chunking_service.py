"""
Chunking service for converting Markdown to semantic RAG chunks.
Implements sentence-window chunking with hierarchical context preservation.
"""

import logging
import re
from markdown_it import MarkdownIt
from langchain_core.documents import Document
from langchain.text_splitter import NLTKTextSplitter


class ChunkingService:
    """
    Service for converting structured Markdown into RAG-optimized chunks.
    Uses semantic blocking and sentence-window strategy.
    """

    def __init__(self, window_size: int = 2, language: str = "spanish"):
        """
        Initialize chunking service.

        Args:
            window_size: Number of sentences before/after for context window
            language: Language for sentence splitting (default: spanish)
        """
        self.window_size = window_size
        self.sentence_splitter = NLTKTextSplitter(language=language)
        logging.info(
            f"Chunking service initialized (window={window_size}, "
            f"language={language})"
        )

    def markdown_to_semantic_blocks(
        self, markdown_text: str
    ) -> list[dict]:
        """
        Parses Markdown and groups into semantic blocks based on headers.

        Args:
            markdown_text: Input markdown content

        Returns:
            List of semantic blocks with path and content
        """
        md = MarkdownIt()
        tokens = md.parse(markdown_text)

        blocks = []
        current_block_content = []
        current_path = []

        # Add dummy token to ensure last block is saved
        tokens.append(
            type("Token", (), {
                "type": "heading_open",
                "tag": "h0",
                "content": "End",
                "level": 0,
            })()
        )

        for i, token in enumerate(tokens):
            if token.type == "heading_open":
                if current_block_content:
                    blocks.append(
                        {
                            "path": (
                                " > ".join(current_path)
                                if current_path
                                else "General Information"
                            ),
                            "content": "".join(current_block_content).strip(),
                        }
                    )

                level = int(token.tag[1]) if token.tag else 99

                header_content = ""
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    header_content = tokens[i + 1].content

                if level == 0:
                    break

                while len(current_path) >= level:
                    current_path.pop()
                current_path.append(header_content)

                current_block_content = [f"{'#' * level} {header_content}\n\n"]

            elif token.type == "paragraph_open":
                if i > 0 and tokens[i - 1].type == "list_item_open":
                    continue
                content = tokens[i + 1].content if i + 1 < len(tokens) else ""
                current_block_content.append(f"{content}\n\n")

            elif token.type == "bullet_list_open":
                current_block_content.append("\n")

            elif token.type == "list_item_open":
                item_content = ""
                for next_token in tokens[i + 1 :]:
                    if (
                        next_token.type == "list_item_close"
                        and next_token.level == token.level
                    ):
                        break
                    if next_token.type == "inline":
                        item_content = next_token.content
                        break

                indent = "    " * (token.level // 2)
                current_block_content.append(f"{indent}- {item_content}\n")

        logging.info(f"Created {len(blocks)} semantic blocks")
        return blocks

    def create_sentence_window_chunks(
        self,
        blocks: list[dict],
        source_file: str,
        medicine_name: str,
    ) -> list[Document]:
        """
        Creates sentence-window chunks from semantic blocks.

        Args:
            blocks: Semantic blocks from markdown_to_semantic_blocks
            source_file: Source filename for metadata
            medicine_name: Medicine name for metadata

        Returns:
            List of LangChain Documents optimized for RAG
        """
        all_chunks = []

        for block in blocks:
            sentences_in_block = []

            content_without_headers = "\n".join(
                [
                    line
                    for line in block["content"].strip().split("\n")
                    if not line.strip().startswith("#")
                ]
            )

            lines = content_without_headers.strip().split("\n")
            paragraph_buffer = []

            for line in lines:
                stripped_line = line.strip()

                if stripped_line.startswith("- "):
                    if paragraph_buffer:
                        full_paragraph = " ".join(paragraph_buffer)
                        sentences_in_block.extend(
                            self.sentence_splitter.split_text(full_paragraph)
                        )
                        paragraph_buffer = []

                    sentences_in_block.append(stripped_line.lstrip("- ").strip())

                elif not stripped_line:
                    if paragraph_buffer:
                        full_paragraph = " ".join(paragraph_buffer)
                        sentences_in_block.extend(
                            self.sentence_splitter.split_text(full_paragraph)
                        )
                        paragraph_buffer = []
                else:
                    paragraph_buffer.append(stripped_line)

            if paragraph_buffer:
                full_paragraph = " ".join(paragraph_buffer)
                sentences_in_block.extend(
                    self.sentence_splitter.split_text(full_paragraph)
                )

            sentences = [s for s in sentences_in_block if s]

            if not sentences:
                continue

            for i, sentence in enumerate(sentences):
                start_index = max(0, i - self.window_size)
                end_index = min(len(sentences), i + self.window_size + 1)
                context_window = " ".join(sentences[start_index:end_index])

                metadata = {
                    "source": source_file,
                    "path": block["path"],
                    "medicine_name": medicine_name,
                    "main_sentence": sentence,
                }

                final_content = f"""---
METADATA:
- Medicine: {metadata['medicine_name']}
- Path: {metadata['path']}
---
CONTEXT:
{context_window}
""".strip()

                all_chunks.append(
                    Document(page_content=final_content, metadata=metadata)
                )

        logging.info(
            f"Created {len(all_chunks)} sentence-window chunks "
            f"for '{medicine_name}'"
        )
        return all_chunks

    def extract_medicine_name(
        self, markdown_text: str, fallback_filename: str
    ) -> str:
        """
        Extracts medicine name from prospectus markdown.

        Args:
            markdown_text: Markdown content
            fallback_filename: Filename to use if extraction fails

        Returns:
            Extracted or fallback medicine name
        """
        try:
            pattern = (
                r"Prospecto: información para el paciente\s*\n\s*(.*?)"
                r"\s*\n\s*Lea todo el prospecto"
            )
            match = re.search(pattern, markdown_text, re.DOTALL | re.IGNORECASE)
            if match:
                medicine_name = " ".join(match.group(1).split()).strip()
                logging.info(f"Extracted medicine name: '{medicine_name}'")
                return medicine_name
        except Exception:
            pass

        name_from_file = (
            fallback_filename.replace(".pdf", "")
            .replace(".md", "")
            .replace("_", " ")
            .replace("-", " ")
        )
        logging.warning(
            f"Could not extract medicine name. Using fallback: '{name_from_file}'"
        )
        return name_from_file

    def standardize_medicine_name(self, raw_name: str) -> str:
        """
        Normalizes medicine name to canonical form.

        Args:
            raw_name: Raw medicine name

        Returns:
            Standardized medicine name
        """
        name = raw_name.lower()

        special_cases = {
            "espidifen": "espidifen",
            "nolotil": "nolotil",
            "sintrom": "sintrom",
            "lexatin": "lexatin",
        }

        for keyword, canonical in special_cases.items():
            if keyword in name:
                return canonical

        if "ibuprofeno" in name and "cinfa" in name:
            return "ibuprofeno cinfa"
        if "ibuprofeno" in name and "kern" in name:
            return "ibuprofeno kern"

        return name.replace("_", " ").replace("-", " ").strip()
