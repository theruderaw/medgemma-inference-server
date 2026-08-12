import io

import pytest
from reportlab.pdfgen import canvas

pytestmark = pytest.mark.analysis
# --------------------------------------------
# Dynamic PDF generator (reusable)
# --------------------------------------------
def create_test_pdf(content: str = "hello world") -> io.BytesIO:
    """Return an in-memory PDF file with the given text."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, content)
    c.save()
    buffer.seek(0)
    return buffer


# --------------------------------------------
# Fixtures
# --------------------------------------------
@pytest.fixture
def stored_document_id(base_url, session):
    """
    Upload a dynamically generated PDF and return its document ID.
    No external files, no CSV storage.
    """
    pdf_buffer = create_test_pdf("analysis test document")
    files = {
        "file": (
            "analysis_sample.pdf",
            pdf_buffer,
            "application/pdf",
        )
    }

    resp = session.post(f"{base_url}/documents/upload", files=files)
    assert resp.status_code == 201, (
        f"Could not upload test PDF: {resp.status_code} {resp.text}"
    )

    document_id = resp.json()["document_id"]
    return document_id


@pytest.fixture
def created_analysis(base_url, session, stored_document_id):
    """
    Trigger analysis on the uploaded document and return the analysis ID.
    """
    doc_id = stored_document_id

    resp = session.post(f"{base_url}/documents/{doc_id}/analysis")
    assert resp.status_code == 202, (
        f"Could not create analysis: {resp.status_code} {resp.text}"
    )

    return resp.json()["analysis_id"]


# --------------------------------------------
# Test classes (example – add your actual tests below)
# --------------------------------------------
class TestGetAnalysis:
    def test_get_existing_analysis(self, base_url, session, created_analysis):
        # Replace with your actual test logic
        # For example, GET /analysis/{analysis_id}
        resp = session.get(f"{base_url}/analysis/{created_analysis}")
        assert resp.status_code == 200
        # ... further assertions


class TestAnalysisStatus:
    def test_get_status_for_existing_analysis(self, base_url, session, created_analysis):
        resp = session.get(f"{base_url}/analysis/{created_analysis}/status")
        assert resp.status_code == 200
        # ... assertions


class TestDeleteAnalysis:
    def test_delete_existing_analysis_returns_204(self, base_url, session, created_analysis):
        resp = session.delete(f"{base_url}/analysis/{created_analysis}")
        assert resp.status_code == 204