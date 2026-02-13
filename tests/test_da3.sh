#!/bin/bash

set -e

trap 'echo "FAIL"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 <num-seeds>"
    exit 1
fi
NUM_SEEDS="$1"

python "$SCRIPT_DIR/test_da3_consistency_before_alternate_attention.py" "$SCRIPT_DIR/samples/cimat_video/" --num-seeds "$NUM_SEEDS"
python "$SCRIPT_DIR/test_da3_vs_da3dino_consistency_check.py" "$SCRIPT_DIR/samples/cimat_video/" --num-seeds "$NUM_SEEDS"
python "$SCRIPT_DIR/test_da3_vs_da3salad_consistency_check.py" "$SCRIPT_DIR/samples/cimat_video/" --num-seeds "$NUM_SEEDS"

echo "PASS"
exit 0
