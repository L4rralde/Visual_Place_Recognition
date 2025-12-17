#Code adapted from https://github.com/serizba/salad/
#@InProceedings{Izquierdo_CVPR_2024_SALAD,
#    author    = {Izquierdo, Sergio and Civera, Javier},
#    title     = {Optimal Transport Aggregation for Visual Place Recognition},
#    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
#    month     = {June},
#    year      = {2024},
#}

from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.validation import get_validation_recalls
# Dataloader
from dataloaders.val.NordlandDataset import NordlandDataset
from dataloaders.val.MapillaryDataset import MSLS
from dataloaders.val.MapillaryTestDataset import MSLSTest #FUTURE
from dataloaders.val.PittsburghDataset import PittsburghDataset
from dataloaders.val.SPEDDataset import SPEDDataset


VAL_DATASETS = ['MSLS', 'MSLS_Test', 'pitts30k_test', 'pitts250k_test', 'Nordland', 'SPED']


def get_val_dataset(dataset_name, transform):
    dataset_name = dataset_name.lower()
    
    if 'nordland' in dataset_name:    
        ds = NordlandDataset(input_transform=transform)

    elif 'msls_test' in dataset_name:
        ds = MSLSTest(input_transform=transform)

    elif 'msls' in dataset_name:
        ds = MSLS(input_transform=transform)

    elif 'pitts' in dataset_name:
        ds = PittsburghDataset(which_ds=dataset_name, input_transform=transform)

    elif 'sped' in dataset_name:
        ds = SPEDDataset(input_transform=transform)
    else:
        raise ValueError
    
    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth


def get_descriptors(model, dataloader, device):
    descriptors = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for batch in tqdm(dataloader, 'Calculating descritptors...'):
                imgs, labels = batch
                output = model(imgs.to(device)).cpu()
                descriptors.append(output)

    return torch.cat(descriptors)


def model_eval(
    model: torch.nn.Module,
    input_transform: Callable, 
    val_datasets: list = VAL_DATASETS,
    verbose: bool = False
):
    model = model.eval()
    model = model.to('cuda')
    torch.backends.cudnn.benchmark = True

    recalls = {}

    for val_name in val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, input_transform)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=32, shuffle=False, pin_memory=True)

        if verbose:
            print(f'Evaluating on {val_name}')
        descriptors = get_descriptors(model, val_loader, 'cuda')
        
        if verbose:
            print(f'Descriptor dimension {descriptors.shape[1]}')
        r_list = descriptors[ : num_references]
        q_list = descriptors[num_references : ]

        if verbose:
            print('total_size', descriptors.shape[0], num_queries + num_references)

        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=False,
        )

        del descriptors
        if verbose:
            print('========> DONE!\n\n')

        recalls[val_name] = preds
    return recalls
