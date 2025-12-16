#"utils" package from https://github.com/serizba/salad
#@InProceedings{Izquierdo_CVPR_2024_SALAD,
#    author    = {Izquierdo, Sergio and Civera, Javier},
#    title     = {Optimal Transport Aggregation for Visual Place Recognition},
#    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#    month     = {June},
#    year      = {2024},
#}

from .losses import get_miner, get_loss
from .validation import get_validation_recalls
