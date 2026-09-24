#!/bin/bash

set -e

trap 'echo "FAIL"; exit 1' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <num-seeds> <vggto-ckpt>"
    exit 1
fi

NUM_SEEDS="$1"
VGGTO_CHKPT="$2"

python "$SCRIPT_DIR/test_vggt_omega_transforms_consistency_check.py" "$SCRIPT_DIR/samples/cimat_video/" --num-seeds "$NUM_SEEDS"
python "$SCRIPT_DIR/test_vggt_omega_vs_vggt_omega_dino_consistency_check.py" "$SCRIPT_DIR/samples/cimat_video/" --num-seeds "$NUM_SEEDS" --ckpt "$VGGTO_CHKPT"
python "$SCRIPT_DIR/test_vggt_omega_salad_vs_vggt_omega_consistency_check.py" "$SCRIPT_DIR/samples/cimat_video/" --num-seeds "$NUM_SEEDS" --ckpt "$VGGTO_CHKPT"

echo "PASS"
exit 0