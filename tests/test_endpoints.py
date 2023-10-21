def test_index(client):
    response = client.get('/')
    assert response.status_code == 200

def test_data_explorer(client):
    response = client.get('/data_explorer')
    assert response.status_code == 200

def test_predict(client):
    response = client.get('/predict')
    assert response.status_code == 200

def test_get_features(client):
    response = client.get('/features')
    assert response.status_code == 200
