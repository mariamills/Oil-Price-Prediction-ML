def test_get_plot(client):
    payload = {
        "selected_features": ["RPI"],
        "plot_type": "line"
    }
    response = client.post('/plot', json=payload)
    data = response.get_json()
    assert response.status_code == 200
    assert "data:image/png;base64," in data["plot_url"]
    assert "data:image/png;base64," in data["legend_url"]
