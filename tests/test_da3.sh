#!/bin/bash

set -e

trap 'echo "FAIL"; exit 1' ERR

python test_da3_consistency_before_alternate_attention.py samples/cimat_video/ --num-seeds 5
python test_da3_vs_da3dino_consistency_check.py samples/cimat_video/ --num-seeds 5
python test_da3_vs_da3salad_consistency_check.py samples/cimat_video/ --num-seeds 5

echo "PASS"
exit 0
