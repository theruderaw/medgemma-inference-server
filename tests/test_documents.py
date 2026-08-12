import io
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw

# Mark the whole module as "documents" tests
pytestmark = pytest.mark.documents


# ------------------------------
# Dynamic test file generators
# ------------------------------
def create_test_pdf(content: str = "hello world") -> io.BytesIO:
    """Return an in-memory PDF file with the given text."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, content)
    c.save()
    buffer.seek(0)
    return buffer


def create_test_image(content: str = "hello world", size=(200, 100)) -> io.BytesIO:
    """Return an in-memory PNG image with the given text."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    # Use a default font; if you have a TTF, you can load it with ImageFont
    draw.text((10, 40), content, fill="black")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


# ------------------------------
# Test Classes
# ------------------------------
class TestUploadDocument:
    def test_upload_returns_201_with_pdf(self, base_url, session):
        """Upload a dynamically generated PDF."""
        pdf_buffer = create_test_pdf("hello from PDF")
        files = {
            "file": (
                "upload_test.pdf",
                pdf_buffer,
                "application/pdf",
            )
        }
        resp = session.post(f"{base_url}/documents/upload", files=files)
        assert resp.status_code == 201

        body = resp.json()
        assert "document_id" in body
        assert body["original_filename"] == "upload_test.pdf"
        assert body["content_type"] == "application/pdf"
        assert body["file_size"] > 0
        assert "created_at" in body

        # No storage – just verify and move on

    def test_upload_returns_201_with_image(self, base_url, session):
        """Upload a dynamically generated PNG image."""
        img_buffer = create_test_image("hello from image")
        files = {
            "file": (
                "upload_test.png",
                img_buffer,
                "image/png",
            )
        }
        resp = session.post(f"{base_url}/documents/upload", files=files)
        assert resp.status_code == 201

        body = resp.json()
        assert "document_id" in body
        assert body["original_filename"] == "upload_test.png"
        assert body["content_type"] == "image/png"
        assert body["file_size"] > 0

    def test_upload_without_file_returns_422(self, base_url, session):
        resp = session.post(f"{base_url}/documents/upload")
        assert resp.status_code == 422


class TestListDocuments:
    def test_list_documents_default_pagination(
        self,
        base_url,
        session,
        uploaded_document,
    ):
        resp = session.get(f"{base_url}/documents")
        assert resp.status_code == 200

        body = resp.json()

        # Handle both plain list and paginated dict
        assert isinstance(body,list)
            
        doc_ids = [d["document_id"] for d in body]
        assert uploaded_document["document_id"] in doc_ids, "Uploaded document not found in list"


    def test_list_documents_with_pagination_params(
        self,
        base_url,
        session,
    ):
        resp = session.get(
            f"{base_url}/documents",
            params={"page_no": 1, "page_size": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Adapt to whatever structure your API returns
        if isinstance(body, dict) and "items" in body:
            assert isinstance(body["items"], list)
        else:
            assert isinstance(body, list)

    def test_list_documents_invalid_pagination_type_returns_422(
        self,
        base_url,
        session,
    ):
        resp = session.get(
            f"{base_url}/documents",
            params={"page_no": "not-a-number"},
        )
        assert resp.status_code == 422


class TestGetDocument:
    def test_get_existing_document(
        self,
        base_url,
        session,
        uploaded_document,
    ):
        doc_id = uploaded_document["document_id"]
        resp = session.get(f"{base_url}/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["document_id"] == doc_id

    def test_get_nonexistent_document_returns_404(
        self,
        base_url,
        session,
        random_uuid,
    ):
        resp = session.get(f"{base_url}/documents/{random_uuid}")
        assert resp.status_code == 404

        body = resp.json()
        # FastAPI uses "detail", not "error"
        assert "detail" in body
        assert body["detail"] == "Document not found"

    def test_get_document_invalid_uuid_returns_422(
        self,
        base_url,
        session,
    ):
        resp = session.get(f"{base_url}/documents/not-a-uuid")
        assert resp.status_code == 422


class TestDeleteDocument:
    def test_delete_existing_document_returns_204(
        self,
        base_url,
        session,
    ):
        # Upload a fresh PDF to delete
        pdf_buffer = create_test_pdf("delete me")
        files = {
            "file": ("to_delete.pdf", pdf_buffer, "application/pdf")
        }
        upload_resp = session.post(f"{base_url}/documents/upload", files=files)
        assert upload_resp.status_code == 201
        doc_id = upload_resp.json()["document_id"]  # key exists now

        delete_resp = session.delete(f"{base_url}/documents/{doc_id}")
        assert delete_resp.status_code == 204

        # Verify it's gone
        get_resp = session.get(f"{base_url}/documents/{doc_id}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent_document_returns_404(
        self,
        base_url,
        session,
        random_uuid,
    ):
        resp = session.delete(f"{base_url}/documents/{random_uuid}")
        assert resp.status_code == 404


@pytest.mark.analysis
class TestDocumentAnalysis:
    def test_trigger_analysis_returns_202(
        self,
        base_url,
        session,
        uploaded_document,
    ):
        doc_id = uploaded_document["document_id"]
        resp = session.post(f"{base_url}/documents/{doc_id}/analysis")
        assert resp.status_code in (202, 409)

        if resp.status_code == 202:
            body = resp.json()
            assert body["document_id"] == doc_id
            assert "analysis_id" in body
            assert body["status"] in (
                "ready",
                "analyzing",
                "chunking",
                "embedding",
                "complete",
                "failed",
                "deleted",
            )

    def test_trigger_analysis_nonexistent_document_returns_404(
        self,
        base_url,
        session,
        random_uuid,
    ):
        resp = session.post(f"{base_url}/documents/{random_uuid}/analysis")
        assert resp.status_code == 404

    def test_list_analyses_for_document(
        self,
        base_url,
        session,
        uploaded_document,
    ):
        doc_id = uploaded_document["document_id"]
        resp = session.get(f"{base_url}/documents/{doc_id}/analyses")
        assert resp.status_code == 200
        body = resp.json()
        # Accept either list or paginated dict
        if isinstance(body, dict) and "items" in body:
            assert isinstance(body["items"], list)
        else:
            assert isinstance(body, list)

    def test_list_analyses_supports_pagination(
        self,
        base_url,
        session,
        uploaded_document,
    ):
        doc_id = uploaded_document["document_id"]
        resp = session.get(
            f"{base_url}/documents/{doc_id}/analyses",
            params={"page_no": 1, "page_size": 5},
        )
        assert resp.status_code == 200

    def test_delete_analyses_for_document(
        self,
        base_url,
        session,
        uploaded_document,
    ):
        doc_id = uploaded_document["document_id"]
        resp = session.delete(f"{base_url}/documents/{doc_id}/analyses")
        assert resp.status_code in (204, 404)

    def test_delete_analyses_nonexistent_document_returns_404(
        self,
        base_url,
        session,
        random_uuid,
    ):
        resp = session.delete(f"{base_url}/documents/{random_uuid}/analyses")
        assert resp.status_code == 404