from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_prediction():

    sample_features = [0.0] * 30

    response = client.post(
        "/predict",
        json={"features": sample_features}
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "fraud_probability" in data