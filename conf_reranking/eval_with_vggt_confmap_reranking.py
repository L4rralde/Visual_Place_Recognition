from copy import copy

import torch
from torch.utils.data import DataLoader
import faiss
import faiss.contrib.torch_utils
import numpy as np
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from eval import get_val_dataset, get_descriptors
from vpr.models.helper import get_transforms
from hubconf import vggt_salad


def main():
    # Load model
    veggiet_salad = vggt_salad('.')
    model = torch.hub.load("serizba/salad", "dinov2_salad")

    device = (
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    model = model.eval().to(device)
    veggiet_salad = veggiet_salad.eval().to(device)

    #Instantiate transform function
    config = {
            'img_size': [322, 322]
        }
    _, input_transform = get_transforms('dino', config)

    #Load dataset
    val_name = 'SPED'
    val_dataset, num_references, num_queries, ground_truth = (
       get_val_dataset(val_name, input_transform)
    )
    val_loader = DataLoader(
            val_dataset,
            num_workers=8,
            batch_size=32,
            shuffle=False,
            pin_memory=True
        )
    val_dataset_clean = copy(val_dataset)
    val_dataset_clean.input_transform = None

    descriptors = get_descriptors(model, val_loader, device)


    r_list = descriptors[ : num_references]
    q_list = descriptors[num_references : ]
    q_list = q_list[:100]

    #Use FAISS to match queries
    embed_size = r_list.shape[1]
    # build index
    faiss_index = faiss.IndexFlatL2(embed_size)
    faiss_index.add(r_list)
    # search for queries in the index
    _, predictions = faiss_index.search(q_list, 10) #TOPK

    normal_matches = 0
    new_matches = 0
    for q_idx, pred in tqdm(enumerate(predictions)):
        q_img, _ = val_dataset_clean[q_idx]
        topk_preds = pred[:5]
        r_imgs = [val_dataset_clean[idx][0] for idx in topk_preds]
        print(topk_preds, ground_truth[q_idx])

        pairwise_confs = veggiet_salad.rerank_by_conf(q_img, r_imgs)
        #pairwise_confs = veggiet_salad.rerank_by_conf_no_pairwise(q_img, r_imgs)
        new_preds = topk_preds[torch.argsort(pairwise_confs, descending=True)]
        print(new_preds, pairwise_confs)
        

        match = np.any(np.in1d(topk_preds[:1], ground_truth[q_idx]))
        if match:
            normal_matches += 1
        else:
            print("SALAD FAILED")
        new_match = np.any(np.in1d(new_preds[:1], ground_truth[q_idx]))
        if new_match:
            new_matches += 1
        else:
            print("VGGT FAILED")
        print('-'*100)


    print(normal_matches, new_matches, q_list.shape)
    

if __name__ == '__main__':
    main()