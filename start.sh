#!/bin/bash
echo "Generating dataset if missing..."
python create_data.py
echo "Starting Gunicorn server..."
gunicorn app:app