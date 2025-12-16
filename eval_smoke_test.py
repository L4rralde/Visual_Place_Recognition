import torch
import torchvision.transforms as v2

from eval import model_eval


def smoke_test():
    transform = v2.Compose([
        v2.Resize((322, 322), antialias=True),
        v2.ToTensor(),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ) #Imagenete
    ])

    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")

    model_eval(model, transform)


if __name__ == '__main__':
    smoke_test()
