#!/bin/bash

set -e

trap 'echo "FAIL"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 <num-seeds>"
    exit 1
fi
NUM_SEEDS="$1"

python "$SCRIPT_DIR/test_vggt_salad_vs_vggt_consistency_check.py" "$SCRIPT_DIR/samples/cimat_video/" --num-seeds "$NUM_SEEDS"
python "$SCRIPT_DIR/test_vggt_transforms_consistency_check.py" "$SCRIPT_DIR/samples/cimat_video/" --num-seeds "$NUM_SEEDS"


echo "PASS"
exit 0
