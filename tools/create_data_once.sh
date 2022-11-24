#!/usr/bin/env bash



PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python tools/create_data.py once --root-path ./data/once --out-dir ./data/once --extra-tag once --version v1.0