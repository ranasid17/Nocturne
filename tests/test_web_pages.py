from web_app import create_app, routes


def test_dashboard_and_local_assets(monkeypatch):
    monkeypatch.setattr(routes, "_available_tickers", lambda: ["AAPL", "UPRO"])
    client = create_app().test_client()
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '<option value="UPRO">' in html
    assert 'data-prediction-url="/api/predictions/run"' in html
    for path in ("css/dashboard.css", "js/dashboard.js", "icons/play.svg", "icons/layers.svg", "icons/refresh-cw.svg"):
        assert client.get(f"/static/{path}").status_code == 200


def test_dashboard_empty_tickers(monkeypatch):
    monkeypatch.setattr(routes, "_available_tickers", lambda: [])
    response = create_app().test_client().get("/")
    assert b"No local ticker history found." in response.data
    assert b'id="ticker"' in response.data


def test_dashboard_ticker_discovery_failure(monkeypatch):
    def fail():
        raise OSError("Private configuration path")

    monkeypatch.setattr(routes, "_available_tickers", fail)
    response = create_app().test_client().get("/")
    assert response.status_code == 200
    assert b"Available tickers could not be loaded." in response.data
    assert b"Private configuration path" not in response.data


def test_dashboard_escapes_ticker_values(monkeypatch):
    monkeypatch.setattr(routes, "_available_tickers", lambda: ['<script>alert(1)</script>'])
    response = create_app().test_client().get("/")
    assert b"<script>alert(1)</script>" not in response.data
