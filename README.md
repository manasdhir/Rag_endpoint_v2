sudo apt install libreoffice redis-server

redis-server --daemonize yes

must have have uv 

then clone the repo using

git clone

uv sync

uv run uvicorn app_new:app --host 0.0.0.0 --port 8000
