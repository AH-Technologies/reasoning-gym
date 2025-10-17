#!/bin/bash
# Helper script to activate the environment
source .venv/bin/activate
source .env
echo "Environment activated!"
echo "You can now run: python scripts/train.py configs/experiments/leg_counting_qwen7b.yaml"
