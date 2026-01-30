#!/bin/bash
set -e

echo "=== Installing packages ==="
pip install --no-cache-dir Flask==3.0.0 flask-cors==4.0.0 gunicorn==21.2.0
pip install --no-cache-dir praw==7.7.1
pip install --no-cache-dir "pandas<2.1" "numpy<1.25"
pip install --no-cache-dir "spacy<3.7"

echo "=== Downloading spaCy model ==="
python -m spacy download en_core_web_sm

echo "=== Verifying spaCy model ==="
python -c "import spacy; spacy.load('en_core_web_sm'); print('Model loaded successfully')"

echo "=== Build complete ==="
