#!/usr/bin/env bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Dev env ready. Activate with: source .venv/bin/activate"