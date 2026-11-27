release: python -m integration.db.init_db
web: uvicorn integration.api:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'
