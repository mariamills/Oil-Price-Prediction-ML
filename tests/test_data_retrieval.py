def test_get_features_data(client):
    response = client.get('/features')
    data = response.get_json()
    assert isinstance(data, list)
    assert "RPI" in data

def test_get_data_success(client):
    payload = {
        "selected_features": ["RPI"]
    }

    response = client.post("/data", json=payload)

    assert response.status_code == 200

    data = response.get_json()
    assert "RPI" in data

