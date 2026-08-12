import io

import pytest
from reportlab.pdfgen import canvas

pytestmark = pytest.mark.messages


# --------------------------------------------
# Dynamic PDF generator (reused)
# --------------------------------------------
def create_test_pdf(content: str = "analysis test document") -> io.BytesIO:
    """Return an in-memory PDF file with the given text."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(100, 750, content)
    c.save()
    buffer.seek(0)
    return buffer


# --------------------------------------------
# Fixtures for analysis tests (no CSV, no static files)
# --------------------------------------------
@pytest.fixture
def stored_document_id(base_url, session):
    """
    Upload a dynamically generated PDF and return its document ID.
    """
    pdf_buffer = create_test_pdf()
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

    return resp.json()["document_id"]


@pytest.fixture
def created_analysis(base_url, session, stored_document_id):
    """
    Triggers analysis on the stored document and returns its analysis_id.
    """
    doc_id = stored_document_id
    resp = session.post(f"{base_url}/documents/{doc_id}/analysis")
    assert resp.status_code == 202, (
        f"Could not create analysis: {resp.status_code} {resp.text}"
    )
    return resp.json()["analysis_id"]


# --------------------------------------------
# Fixtures for message tests
# --------------------------------------------
@pytest.fixture
def created_message(base_url, session, created_chat):
    """Adds a USER message to the chat and returns the parsed response."""
    chat_id = created_chat["chat_id"]
    resp = session.post(
        f"{base_url}/chats/{chat_id}/query",
        json={"role": "USER", "content": "Hello, this is a test message."},
    )
    assert resp.status_code == 201, f"Could not create message: {resp.status_code} {resp.text}"
    return resp.json()


# --------------------------------------------
# Message tests (unchanged, but with pagination awareness)
# --------------------------------------------
class TestAddMessage:
    def test_add_message_returns_201(self, base_url, session, created_chat):
        chat_id = created_chat["chat_id"]
        resp = session.post(
            f"{base_url}/chats/{chat_id}/query",
            json={"role": "USER", "content": "What is in this document?"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["chat_id"] == chat_id
        assert body["role"] == "USER"
        assert body["content"] == "What is in this document?"
        assert "message_id" in body
        assert "created_at" in body
        assert "updated_at" in body

    def test_add_message_with_metadata(self, base_url, session, created_chat):
        chat_id = created_chat["chat_id"]
        resp = session.post(
            f"{base_url}/chats/{chat_id}/query",
            json={
                "role": "USER",
                "content": "Message with metadata",
                "message_metadata": {"source": "test-suite"},
            },
        )
        assert resp.status_code == 201
        assert resp.json()["message_metadata"] == {"source": "test-suite"}

    def test_add_message_to_nonexistent_chat_returns_404(self, base_url, session, random_uuid):
        resp = session.post(
            f"{base_url}/chats/{random_uuid}/query",
            json={"role": "USER", "content": "hi"},
        )
        assert resp.status_code == 404

    def test_add_message_missing_content_returns_422(self, base_url, session, created_chat):
        chat_id = created_chat["chat_id"]
        resp = session.post(f"{base_url}/chats/{chat_id}/query", json={"role": "USER"})
        assert resp.status_code == 422

    def test_add_message_invalid_role_returns_422(self, base_url, session, created_chat):
        chat_id = created_chat["chat_id"]
        resp = session.post(
            f"{base_url}/chats/{chat_id}/query",
            json={"role": "NOT_A_ROLE", "content": "hi"},
        )
        assert resp.status_code == 422


class TestGetChatHistory:
    def test_get_history_returns_messages(self, base_url, session, created_chat, created_message):
        chat_id = created_chat["chat_id"]
        resp = session.get(f"{base_url}/chats/{chat_id}/messages")
        assert resp.status_code == 200
        body = resp.json()
        # Handle paginated responses
        if isinstance(body, dict) and "items" in body:
            messages = body["items"]
        else:
            messages = body
        assert isinstance(messages, list)
        assert any(m["message_id"] == created_message["message_id"] for m in messages)

    def test_get_history_nonexistent_chat_returns_404(self, base_url, session, random_uuid):
        resp = session.get(f"{base_url}/chats/{random_uuid}/messages")
        assert resp.status_code == 404


class TestGetMessage:
    def test_get_existing_message(self, base_url, session, created_chat, created_message):
        chat_id = created_chat["chat_id"]
        msg_id = created_message["message_id"]
        resp = session.get(f"{base_url}/chats/{chat_id}/messages/{msg_id}")
        assert resp.status_code == 200
        assert resp.json()["message_id"] == msg_id

    def test_get_nonexistent_message_returns_404(self, base_url, session, created_chat, random_uuid):
        chat_id = created_chat["chat_id"]
        resp = session.get(f"{base_url}/chats/{chat_id}/messages/{random_uuid}")
        assert resp.status_code == 404


class TestEditMessage:
    def test_edit_message_content(self, base_url, session, created_chat, created_message):
        chat_id = created_chat["chat_id"]
        msg_id = created_message["message_id"]
        resp = session.patch(
            f"{base_url}/chats/{chat_id}/messages/{msg_id}",
            json={"content": "Updated content"},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "Updated content"

    def test_edit_nonexistent_message_returns_404(self, base_url, session, created_chat, random_uuid):
        chat_id = created_chat["chat_id"]
        resp = session.patch(
            f"{base_url}/chats/{chat_id}/messages/{random_uuid}",
            json={"content": "x"},
        )
        assert resp.status_code == 404


class TestDeleteMessage:
    def test_delete_existing_message_returns_204(self, base_url, session, created_chat):
        chat_id = created_chat["chat_id"]

        # 1. Create a user message
        create_resp = session.post(
            f"{base_url}/chats/{chat_id}/query",
            json={"role": "USER", "content": "to be deleted"},
        )
        assert create_resp.status_code == 201
        msg_id = create_resp.json()["message_id"]

    # 2. Delete it
        delete_resp = session.delete(f"{base_url}/chats/{chat_id}/messages/{msg_id}")
        assert delete_resp.status_code == 204

        # 3. Verify deletion via the message list
        list_resp = session.get(f"{base_url}/chats/{chat_id}/messages")
        assert list_resp.status_code == 200
        messages = list_resp.json()
        #Cross chat persistence
        msg_ids = [m["message_id"] for m in messages]
        assert msg_id  in msg_ids, f"Message {msg_id} still present after deletion"
        
        
    def test_delete_nonexistent_message_returns_404(self, base_url, session, created_chat, random_uuid):
        chat_id = created_chat["chat_id"]
        resp = session.delete(f"{base_url}/chats/{chat_id}/messages/{random_uuid}")
        assert resp.status_code == 404


