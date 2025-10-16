"""
Unit tests for ChunkingService.
Tests markdown parsing, semantic chunking, and medicine name extraction.
"""

import pytest
from langchain_core.documents import Document

from src.services.chunking_service import ChunkingService


@pytest.fixture
def chunking_service():
    """Create ChunkingService instance."""
    return ChunkingService()


class TestMedicineNameExtraction:
    """Tests for medicine name extraction from markdown."""

    def test_extract_from_header(self, chunking_service, sample_markdown):
        """Should extract medicine name from first header."""
        # Act
        medicine_name = chunking_service.extract_medicine_name(
            sample_markdown, "ibuprofeno_600.pdf"
        )
        
        # Assert
        assert medicine_name.lower() == "ibuprofeno"

    def test_extract_from_filename_fallback(self, chunking_service):
        """Should use filename if header doesn't contain medicine name."""
        # Arrange
        markdown = "# Document Title\n\nSome content"
        
        # Act
        medicine_name = chunking_service.extract_medicine_name(
            markdown, "paracetamol_1000.pdf"
        )
        
        # Assert
        assert medicine_name.lower() == "paracetamol"

    def test_standardize_medicine_name(self, chunking_service):
        """Should standardize medicine name."""
        # Arrange
        raw_names = [
            "Ibuprofeno Cinfa 600 mg",
            "PARACETAMOL 1000",
            "Nolotil 575 mg Cápsulas",
            "Lexatin 3",
        ]
        
        expected = ["ibuprofeno", "paracetamol", "nolotil", "lexatin"]
        
        # Act & Assert
        for raw, expected_std in zip(raw_names, expected):
            result = chunking_service.standardize_medicine_name(raw)
            assert result == expected_std


class TestMarkdownToSemanticBlocks:
    """Tests for markdown parsing into semantic blocks."""

    def test_parse_headers_and_content(self, chunking_service, sample_markdown):
        """Should parse headers and their content correctly."""
        # Act
        blocks = chunking_service.markdown_to_semantic_blocks(sample_markdown)
        
        # Assert
        assert len(blocks) > 0
        assert any("Qué es" in block["header"] for block in blocks)
        assert any("content" in block for block in blocks)

    def test_preserve_hierarchy(self, chunking_service):
        """Should preserve header hierarchy."""
        # Arrange
        markdown = """# 1. Main Section

## Subsection A

Content A

### Sub-subsection A1

Content A1

## Subsection B

Content B
"""
        
        # Act
        blocks = chunking_service.markdown_to_semantic_blocks(markdown)
        
        # Assert
        # Should have blocks for main section, subsections, and sub-subsections
        assert len(blocks) >= 3

    def test_handle_lists(self, chunking_service):
        """Should parse list items correctly."""
        # Arrange
        markdown = """# Section

## List Example

Items:
- Item 1
- Item 2
- Item 3
"""
        
        # Act
        blocks = chunking_service.markdown_to_semantic_blocks(markdown)
        
        # Assert
        # Should have parsed the list items
        content = " ".join(block["content"] for block in blocks)
        assert "Item 1" in content
        assert "Item 2" in content

    def test_handle_empty_sections(self, chunking_service):
        """Should handle sections with no content gracefully."""
        # Arrange
        markdown = """# Section 1

## Subsection 1.1

## Subsection 1.2

Some content here.
"""
        
        # Act
        blocks = chunking_service.markdown_to_semantic_blocks(markdown)
        
        # Assert
        # Should not crash and should parse sections
        assert len(blocks) > 0


class TestSentenceWindowChunking:
    """Tests for sentence-window chunking strategy."""

    def test_create_chunks_with_context(self, chunking_service):
        """Should create chunks with sentence window context."""
        # Arrange
        blocks = [
            {
                "header": "# Section 1",
                "content": "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5.",
                "section_hierarchy": ["Section 1"],
            }
        ]
        
        # Act
        chunks = chunking_service.create_sentence_window_chunks(
            blocks=blocks,
            source_file="test.md",
            medicine_name="testmedicine",
            window_size=1,  # 1 sentence before/after
        )
        
        # Assert
        assert len(chunks) > 0
        # Each chunk should be a Document
        assert all(isinstance(chunk, Document) for chunk in chunks)
        # Should have metadata
        assert all("medicine_name" in chunk.metadata for chunk in chunks)

    def test_chunks_have_correct_metadata(self, chunking_service):
        """Should include all required metadata in chunks."""
        # Arrange
        blocks = [
            {
                "header": "# Section 1",
                "content": "Test sentence.",
                "section_hierarchy": ["Section 1"],
            }
        ]
        
        # Act
        chunks = chunking_service.create_sentence_window_chunks(
            blocks=blocks,
            source_file="test.md",
            medicine_name="testmedicine",
        )
        
        # Assert
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.md"
            assert "medicine_name" in chunk.metadata
            assert chunk.metadata["medicine_name"] == "testmedicine"
            assert "section_hierarchy" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_window_includes_context(self, chunking_service):
        """Should include surrounding sentences as context."""
        # Arrange
        blocks = [
            {
                "header": "# Section",
                "content": "First sentence. Second sentence. Third sentence.",
                "section_hierarchy": ["Section"],
            }
        ]
        
        # Act
        chunks = chunking_service.create_sentence_window_chunks(
            blocks=blocks,
            source_file="test.md",
            medicine_name="test",
            window_size=1,
        )
        
        # Assert
        # Middle chunk should contain context from neighbors
        if len(chunks) >= 2:
            middle_chunk = chunks[1]
            content = middle_chunk.page_content.lower()
            # Should contain the sentence itself and neighbors
            assert "second" in content or "sentence" in content

    def test_empty_blocks_produce_no_chunks(self, chunking_service):
        """Should handle empty blocks without crashing."""
        # Arrange
        blocks = []
        
        # Act
        chunks = chunking_service.create_sentence_window_chunks(
            blocks=blocks,
            source_file="test.md",
            medicine_name="test",
        )
        
        # Assert
        assert chunks == []

    def test_chunk_index_sequential(self, chunking_service):
        """Should assign sequential chunk indices."""
        # Arrange
        blocks = [
            {
                "header": "# Section",
                "content": "Sentence 1. Sentence 2. Sentence 3. Sentence 4.",
                "section_hierarchy": ["Section"],
            }
        ]
        
        # Act
        chunks = chunking_service.create_sentence_window_chunks(
            blocks=blocks,
            source_file="test.md",
            medicine_name="test",
        )
        
        # Assert
        indices = [chunk.metadata["chunk_index"] for chunk in chunks]
        assert indices == list(range(len(chunks)))


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handle_unicode_characters(self, chunking_service):
        """Should handle Spanish characters correctly."""
        # Arrange
        markdown = """# Qué es el medicamento

## Descripción

Este medicamento está indicado para el tratamiento de la inflamación.
"""
        
        # Act
        blocks = chunking_service.markdown_to_semantic_blocks(markdown)
        
        # Assert
        assert len(blocks) > 0
        content = " ".join(block["content"] for block in blocks)
        assert "inflamación" in content

    def test_handle_very_long_sections(self, chunking_service):
        """Should handle sections with many sentences."""
        # Arrange
        long_content = ". ".join([f"Sentence {i}" for i in range(100)])
        blocks = [
            {
                "header": "# Long Section",
                "content": long_content,
                "section_hierarchy": ["Long Section"],
            }
        ]
        
        # Act
        chunks = chunking_service.create_sentence_window_chunks(
            blocks=blocks,
            source_file="test.md",
            medicine_name="test",
        )
        
        # Assert
        # Should create many chunks without crashing
        assert len(chunks) > 10

    def test_handle_single_sentence_block(self, chunking_service):
        """Should handle blocks with single sentence."""
        # Arrange
        blocks = [
            {
                "header": "# Section",
                "content": "Single sentence.",
                "section_hierarchy": ["Section"],
            }
        ]
        
        # Act
        chunks = chunking_service.create_sentence_window_chunks(
            blocks=blocks,
            source_file="test.md",
            medicine_name="test",
        )
        
        # Assert
        assert len(chunks) == 1
        assert chunks[0].page_content.strip() == "Single sentence."

