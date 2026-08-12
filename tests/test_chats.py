import pytest


pytestmark = pytest.mark.chats


class TestCreateChat:
    def test_create_chat_with_title(self, base_url, session):
        resp = session.post(
            f"{base_url}/chats",
            json={"title": "My Chat"},
        )

        assert resp.status_code == 201

        body = resp.json()

        assert "chat_id" in body
        assert body["title"] == "My Chat"

        session.delete(
            f"{base_url}/chats/{body['chat_id']}"
        )

    def test_create_chat_without_title(self, base_url, session):
        resp = session.post(
            f"{base_url}/chats",
            json={},
        )

        assert resp.status_code == 201

        body = resp.json()

        assert "chat_id" in body

        session.delete(
            f"{base_url}/chats/{body['chat_id']}"
        )


class TestListChats:
    def test_list_chats_default_pagination(
        self,
        base_url,
        session,
        created_chat,
    ):
        resp = session.get(
            f"{base_url}/chats"
        )

        assert resp.status_code == 200

        body = resp.json()

        assert isinstance(body, list)
        assert any(
            c["chat_id"] == created_chat["chat_id"]
            for c in body
        )

    def test_list_chats_with_pagination_params(
        self,
        base_url,
        session,
    ):
        resp = session.get(
            f"{base_url}/chats",
            params={
                "page_no": 1,
                "page_size": 5,
            },
        )

        assert resp.status_code == 200


class TestGetChat:
    def test_get_existing_chat(
        self,
        base_url,
        session,
        created_chat,
    ):
        chat_id = created_chat["chat_id"]

        resp = session.get(
            f"{base_url}/chats/{chat_id}"
        )

        assert resp.status_code == 200
        assert resp.json()["chat_id"] == chat_id

    def test_get_nonexistent_chat_returns_404(
        self,
        base_url,
        session,
        random_uuid,
    ):
        resp = session.get(
            f"{base_url}/chats/{random_uuid}"
        )

        assert resp.status_code == 404


class TestPatchChat:
    def test_patch_chat_title(
        self,
        base_url,
        session,
        created_chat,
    ):
        chat_id = created_chat["chat_id"]

        resp = session.patch(
            f"{base_url}/chats/{chat_id}",
            json={"title": "Renamed Chat"},
        )

        assert resp.status_code == 200
        assert resp.json()["title"] == "Renamed Chat"

    def test_patch_nonexistent_chat_returns_404(
        self,
        base_url,
        session,
        random_uuid,
    ):
        resp = session.patch(
            f"{base_url}/chats/{random_uuid}",
            json={"title": "x"},
        )

        assert resp.status_code == 404


class TestDeleteChat:
    def test_delete_existing_chat_returns_204(
        self,
        base_url,
        session,
    ):
        create_resp = session.post(
            f"{base_url}/chats",
            json={"title": "Temp"},
        )

        assert create_resp.status_code == 201

        chat_id = create_resp.json()["chat_id"]

        delete_resp = session.delete(
            f"{base_url}/chats/{chat_id}"
        )

        assert delete_resp.status_code == 204

        get_resp = session.get(
            f"{base_url}/chats/{chat_id}"
        )

        assert get_resp.status_code == 404

    def test_delete_nonexistent_chat_returns_404(
        self,
        base_url,
        session,
        random_uuid,
    ):
        resp = session.delete(
            f"{base_url}/chats/{random_uuid}"
        )

        assert resp.status_code == 404


class TestChatDocumentAssignment:
    def test_assign_document_to_chat(
        self,
        base_url,
        session,
        created_chat,
        uploaded_document,
    ):
        chat_id = created_chat["chat_id"]
        doc_id = uploaded_document["document_id"]

        resp = session.post(
            f"{base_url}/chats/{chat_id}/document/{doc_id}"
        )

        assert resp.status_code == 202

        body = resp.json()

        assert body["chat_id"] == chat_id
        assert body["document_id"] == doc_id

        session.delete(
            f"{base_url}/chats/{chat_id}/document/{doc_id}"
        )

    def test_assign_document_nonexistent_chat_returns_404(
        self,
        base_url,
        session,
        random_uuid,
        uploaded_document,
    ):
        doc_id = uploaded_document["document_id"]

        resp = session.post(
            f"{base_url}/chats/{random_uuid}/document/{doc_id}"
        )

        assert resp.status_code == 404

    def test_list_chat_documents(
        self,
        base_url,
        session,
        created_chat,
        uploaded_document,
    ):
        chat_id = created_chat["chat_id"]
        doc_id = uploaded_document["document_id"]

        assign_resp = session.post(
            f"{base_url}/chats/{chat_id}/document/{doc_id}"
        )

        assert assign_resp.status_code == 202

        resp = session.get(
            f"{base_url}/chats/{chat_id}/documents"
        )

        assert resp.status_code == 200

        body = resp.json()

        assert isinstance(body, list)
        assert any(
            d["document_id"] == doc_id
            for d in body
        )

        session.delete(
            f"{base_url}/chats/{chat_id}/document/{doc_id}"
        )

    def test_list_documents_nonexistent_chat_returns_404(
        self,
        base_url,
        session,
        random_uuid,
    ):
        resp = session.get(
            f"{base_url}/chats/{random_uuid}/documents"
        )

        assert resp.status_code == 404

    def test_remove_document_from_chat(
        self,
        base_url,
        session,
        created_chat,
        uploaded_document,
    ):
        chat_id = created_chat["chat_id"]
        doc_id = uploaded_document["document_id"]

        assign_resp = session.post(
            f"{base_url}/chats/{chat_id}/document/{doc_id}"
        )

        assert assign_resp.status_code == 202

        resp = session.delete(
            f"{base_url}/chats/{chat_id}/document/{doc_id}"
        )

        assert resp.status_code == 204

    def test_remove_document_not_assigned_returns_404(
        self,
        base_url,
        session,
        created_chat,
        uploaded_document,
    ):
        chat_id = created_chat["chat_id"]
        doc_id = uploaded_document["document_id"]

        resp = session.delete(
            f"{base_url}/chats/{chat_id}/document/{doc_id}"
        )

        assert resp.status_code == 404