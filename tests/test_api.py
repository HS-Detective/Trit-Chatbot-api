import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    """/api/chat/health 엔드포인트가 정상적으로 응답하는지 테스트합니다."""
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["ok"] is True
    assert "faq" in json_response["features"]
    assert "nav" in json_response["features"]

def test_ask_nav_feature(mocker):
    """/api/chat/ask 엔드포인트가 nav 기능으로 요청을 받고, NavChatBot.ask를 호출하는지 테스트합니다."""
    mock_response = {
        "answer": "모의 답변입니다.",
        "sources": [{"source": "test.csv", "page": 1}],
        "chat_history": [
            {"role": "human", "content": "테스트 질문"},
            {"role": "ai", "content": "모의 답변입니다."}
        ]
    }
    mocker.patch("routers.chat.BOTS['nav'].ask", return_value=mock_response)

    request_body = {
        "question": "테스트 질문",
        "feature_id": "nav",
        "chat_history": []
    }
    response = client.post("/ask", json=request_body)

    assert response.status_code == 200
    assert response.json() == mock_response