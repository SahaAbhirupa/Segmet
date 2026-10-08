#!/bin/sh
# GPU queue (CNNs) and CPU queue (classical baselines) run side by side; they write disjoint files.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
( for e in main noise scale; do $PY -u run_study.py --exp $e --models cnn_reg segnet_reg segmet; done ) > results/logs/cnn.log 2>&1 &
( for e in main noise scale; do $PY -u run_study.py --exp $e --models majority lin_reg lin_cls; done ) > results/logs/linear.log 2>&1 &
wait
