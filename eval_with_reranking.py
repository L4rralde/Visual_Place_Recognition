import argparse

import torch
from torch.utils.data import DataLoader
import faiss
import faiss.contrib.torch_utils
import numpy as np
from tqdm import tqdm

from eval import get_val_dataset, get_descriptors
from vpr.reranking import Reranker
from vpr.models.helper import get_transforms


def main():
    # Load model
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    #model = VPRModel.from_lightning_log(args.log_path)
    device = (
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    model = model.eval().to(device)

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

    descriptors = get_descriptors(model, val_loader, device)


    r_list = descriptors[ : num_references]
    q_list = descriptors[num_references : ]

    #Use FAISS to match queries
    embed_size = r_list.shape[1]
    # build index
    faiss_index = faiss.IndexFlatL2(embed_size)
    faiss_index.add(r_list)
    # search for queries in the index
    _, predictions = faiss_index.search(q_list, 10) #TOPK

    #Initialize reranker
    reranker = Reranker(val_dataset)
    reranker.extract_local_features()

    normal_matches = 0
    new_matches = 0
    for q_idx, pred in tqdm(enumerate(predictions)):
        new_idcs = reranker.rerank(q_idx, pred[:5])
        #print(pred[:5], ground_truth[q_idx])
        #print(new_idcs)
        #print('-'*100)

        new_preds = pred[new_idcs['permutation']]
        #print(pred, ground_truth[q_idx])

        match = np.any(np.in1d(pred[:1], ground_truth[q_idx]))
        if match:
            normal_matches += 1
        new_match = np.any(np.in1d(new_preds[:1], ground_truth[q_idx]))
        if new_match:
            new_matches += 1
    
    print(normal_matches, new_matches, q_list.shape)
    
if __name__ == '__main__':
    main()
