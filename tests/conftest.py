"""
Shared pytest fixtures for the API test suite.

The suite talks to a *running* instance of the API over HTTP. Point it at
your server with the BASE_URL environment variable, e.g.:

    BASE_URL=http://localhost:8000 pytest

If BASE_URL is not set, it defaults to http://localhost:8000.
"""
import io
import os
import uuid
from pathlib import Path

import pytest
import requests
from reportlab.pdfgen import canvas

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/api/v1").rstrip("/")


# --------------------------------------------
# Helper: generate a PDF in memory
# --------------------------------------------
def create_test_pdf(content: str = "test document") -> io.BytesIO:
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
@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def session():
    """A single requests.Session reused across the whole run."""
    with requests.Session() as s:
        yield s


@pytest.fixture
def uploaded_document(base_url, session):
    """
    Uploads a dynamically generated PDF and returns its metadata.
    The document is deleted after the test finishes.
    """
    pdf_buffer = create_test_pdf("uploaded from fixture")
    files = {
        "file": (
            "fixture_sample.pdf",
            pdf_buffer,
            "application/pdf",
        )
    }

    resp = session.post(f"{base_url}/documents/upload", files=files)
    assert resp.status_code == 201, (
        f"Could not upload test PDF: {resp.status_code} {resp.text}"
    )

    doc_data = resp.json()
    doc_id = doc_data["document_id"]

    yield doc_data

    # Teardown: delete the uploaded document
    session.delete(f"{base_url}/documents/{doc_id}")


@pytest.fixture
def created_chat(base_url, session):
    """Creates a chat session and yields the parsed ChatRead response."""
    resp = session.post(f"{base_url}/chats", json={"title": "Test Chat"})
    assert resp.status_code == 201, f"Chat creation failed: {resp.status_code} {resp.text}"
    chat = resp.json()

    yield chat

    # Teardown: delete the chat
    session.delete(f"{base_url}/chats/{chat['chat_id']}")


@pytest.fixture
def random_uuid():
    """A random UUID string, useful for exercising 404 paths."""
    return str(uuid.uuid4())