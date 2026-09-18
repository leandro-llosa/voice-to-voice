#!/bin/sh
# Bootstrap the venv on first run, then start the app on http://127.0.0.1:7860
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    uv venv .venv --python 3.12
    uv pip install -p .venv/bin/python -r requirements.txt
fi
exec .venv/bin/python app.py
