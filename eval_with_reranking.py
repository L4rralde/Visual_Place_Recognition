import argparse

import torch
from torch.utils.data import DataLoader
import faiss
import faiss.contrib.torch_utils
import numpy as np


from vpr_model import VPRModel
from eval_trained_vpr_model import Transforms
from eval import get_val_dataset, get_descriptors


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str)
    parser.add_argument('--img-size', type=int, nargs='+')

    #parser.add_argument('--yaml', type=str, default='') #FUTURE
    args = parser.parse_args()
    assert len(args.img_size) < 3, "Expected one or two numbers for image size"

    return args


def main():
    args = parse_args()

    # Load model
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    #model = VPRModel.from_lightning_log(args.log_path)
    device = (
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    model = model.eval().to(device)

    #Instantiate transform function
    img_size = args.img_size
    for size in img_size:
        assert size % model.backbone.PATCH_SIZE == 0, "Img size not divisible by patch size"
    if len(img_size) == 1:
        img_size = img_size[0]
    input_transform = Transforms.get_transform(
        model.encoder_arch,
        img_size
    )

    #Load dataset
    val_name = 'pitts30k_test'
    val_dataset, num_references, num_queries, ground_truth = (
       get_val_dataset(val_name, input_transform)
    )
    val_loader = DataLoader(
            val_dataset,
            num_workers=8,
            batch_size=1,
            shuffle=False,
            pin_memory=True
        )

    #Get descriptors / global features
    descriptors = get_descriptors(model, val_loader, device)

    r_list = descriptors[ : num_references]
    q_list = descriptors[num_references : ]

    #Use FAISS to match queries
    embed_size = r_list.shape[1]
    # build index
    faiss_index = faiss.IndexFlatL2(embed_size)
    # search for queries in the index
    _, predictions = faiss_index.search(q_list, max(10)) #TOPK

    for q_idx, pred in enumerate(predictions):
        print(pred[:5], ground_truth[q_idx])

        match = np.any(np.in1d(pred[:5], ground_truth[q_idx]))
        #
        #Reranking:
        # 1. Need to access to individual images.
        # 2. pass query image and topk matches to reranker
        # 3. Sort according new ranks