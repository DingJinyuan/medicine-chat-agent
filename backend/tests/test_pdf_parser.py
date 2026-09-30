"""L1 PDF 解析测试(离线,用 fitz 生成临时 PDF)。"""
import fitz

from src.ingestion.pdf_parser import PDFParser


def _create_pdf(path, text="Clinical guideline: treat diabetes with metformin."):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


class TestPDFParser:
    def test_parse_file_returns_document(self, tmp_path):
        pdf = tmp_path / "diabetes_guideline.pdf"
        _create_pdf(pdf)
        parser = PDFParser(pdf_dir=str(tmp_path / "guidelines"))
        result = parser.parse_file(str(pdf))
        assert result is not None
        assert result.doc_id == "pdf_diabetes_guideline"
        assert result.title == "Diabetes Guideline"
        assert result.page_count == 1
        assert result.source == "pdf"
        assert "[Page 1]" in result.content
        assert "metformin" in result.content

    def test_parse_file_nonexistent_returns_none(self, tmp_path):
        parser = PDFParser(pdf_dir=str(tmp_path / "guidelines"))
        assert parser.parse_file(str(tmp_path / "missing.pdf")) is None

    def test_parse_directory_empty_returns_empty(self, tmp_path):
        parser = PDFParser(pdf_dir=str(tmp_path / "empty"))
        assert parser.parse_directory() == []

    def test_title_generation_from_filename(self, tmp_path):
        pdf = tmp_path / "heart-failure_treatment.pdf"
        _create_pdf(pdf)
        parser = PDFParser(pdf_dir=str(tmp_path / "guidelines"))
        result = parser.parse_file(str(pdf))
        assert result.title == "Heart Failure Treatment"
