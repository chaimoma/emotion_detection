from fastapi.testclient import TestClient
from app.main import app 

client = TestClient(app)
def test_get_history_format(): 
    response = client.get("/history")
    assert response.status_code == 200, "API call to /history failed"
    data = response.json()
    assert isinstance(data, list), "Response data is not a list"
    #check the keys
    if len(data) > 0:
        first_item = data[0]
        assert "id" in first_item
        assert "emotion" in first_item
        assert "confidence" in first_item
        assert "created_at" in first_item
