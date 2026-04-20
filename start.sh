#!/bin/bash
python create_data.py
gunicorn app:app