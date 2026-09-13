import torchvision.transforms as transforms

import datasets.omniglot as om
from datasets.cifar100_continual import load_cifar100


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, train=True, path=None, background=True, all=False):

        if name == "omniglot":
            train_transform = transforms.Compose(
                [transforms.Resize((84, 84)),
                 transforms.ToTensor()])
            if path is None:
                return om.Omniglot("../data/omni", background=background, download=True, train=train,
                                   transform=train_transform, all=all)
            else:


                return om.Omniglot(
                    path,
                    background=background,
                    download=True,
                    train=train,
                    transform=train_transform,
                    all=all,
                )

        elif name.lower() == "cifar100":
            root = path if path is not None else "../data/cifar100"
            return load_cifar100(
                root=root,
                train=train,
                image_size=28,
                augment=train,
                download_if_missing=True,
            )
        else:
            print("Unsupported Dataset")
            assert False
