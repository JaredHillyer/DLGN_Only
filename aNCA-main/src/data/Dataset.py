from typing import Any, Tuple
from collections import OrderedDict
import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import v2
import torch.nn.functional as F
from torchvision.datasets import VisionDataset
import os
from utils import show_distribution_all
from torchvision.datasets.utils import _decompress, download_file_from_google_drive, verify_str_arg
import pathlib
import h5py

class WBC_Dataset(data.Dataset):
    def __init__(self,image_paths,labels,resizeW, resizeH, augment=False, dataset="AML"):
        self.num_classes=13
        self.image_paths=image_paths
        self.labels=labels
        self.resizeW=resizeW
        self.resizeH=resizeH
        self.augment=augment
        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5),
            # v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25, hue=0.1),
            # v2.RandomAffine(10, (0.05, 0.05), fill=(255, 255, 255)),
            ])
        if dataset =="AML":
            self.norm = v2.Compose([
                v2.Resize((self.resizeH, self.resizeW)),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True), 
                v2.Normalize(mean=[0.82069695, 0.7281261, 0.836143],std=[0.16157213, 0.2490039, 0.09052657])])
        elif dataset =="PBC":
            self.norm = v2.Compose([
                v2.Resize((self.resizeH, self.resizeW)),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True), 
                v2.Normalize(mean=[0.8746204, 0.7487587, 0.7203138],std=[0.15061052, 0.17617777, 0.07467376])])
        elif dataset =="MLL":
            self.norm = v2.Compose([
                v2.Resize((self.resizeH, self.resizeW)),
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True), 
                v2.Normalize(mean=[0.74053776, 0.6514114, 0.7785342],std=[0.18301032, 0.24672535, 0.16100405])])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.augment:
            image = self.transforms(image)
            
        image=self.norm(image)
        label=torch.zeros(self.num_classes)
        label[self.labels[idx]]=1

        return image.permute(1,2,0), label

class Malaria(data.Dataset):
    def __init__(self,image_paths,labels,resizeW, resizeH, augment=False):
        self.num_classes=2
        self.image_paths=image_paths
        self.labels=labels
        self.resizeW=resizeW
        self.resizeH=resizeH
        self.augment=augment
        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25, hue=0.1),
            v2.RandomAffine(10, (0.05, 0.05), fill=(255, 255, 255)),
            ])
        self.norm = v2.Compose([
            v2.Resize((self.resizeH, self.resizeW)),
            v2.ToImage(), # (C, H, W)
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.5286651253700256, 0.42176899313926697, 0.4479588568210602],std=[0.34576717019081116, 0.2789415419101715, 0.29265928268432617])])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.augment:
            image = self.transforms(image)
            
        image=self.norm(image)
        label=torch.zeros(self.num_classes)
        label[self.labels[idx]]=1

        return image.permute(1,2,0), label

class SIPAKMED(data.Dataset):
    def __init__(self,image_paths,labels,resizeW, resizeH, augment=False):
        self.num_classes=5
        self.image_paths=image_paths
        self.labels=labels
        self.resizeW=resizeW
        self.resizeH=resizeH
        self.augment=augment
        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25, hue=0.1),
            v2.RandomAffine(10, (0.05, 0.05), fill=(255, 255, 255)),
            ])
        self.norm = v2.Compose([
            v2.Resize((self.resizeH, self.resizeW)),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.5758177042007446, 0.5742955207824707, 0.6884000301361084],std=[0.2185823619365692, 0.19480502605438232, 0.19809827208518982])])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.augment:
            image = self.transforms(image)
            
        image=self.norm(image)
        label=torch.zeros(self.num_classes)
        label[self.labels[idx]]=1

        return image.permute(1,2,0), label
    
class Urine(data.Dataset):
    def __init__(self,image_paths,labels, resizeW, resizeH, augment=False):
        self.num_classes=8
        self.image_paths=image_paths
        self.labels=labels
        self.resizeW=resizeW
        self.resizeH=resizeH
        self.augment=augment
        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25, hue=0.1),
            v2.RandomAffine(10, (0.05, 0.05), fill=(255, 255, 255)),
            ])
        self.norm = v2.Compose([
            v2.Resize((self.resizeH, self.resizeW)),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.6369247436523438, 0.7650187611579895, 0.7165603637695312],std=[0.05541696399450302, 0.05346846953034401, 0.056218646466732025])])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.augment:
            image = self.transforms(image)
            
        image=self.norm(image)
        label=torch.zeros(self.num_classes)
        label[self.labels[idx]]=1

        return image.permute(1,2,0), label

class CIFAR10WithOneHot(torch.utils.data.Dataset):
    def __init__(self, dataset, resizeW, resizeH, augment=False):
        self.dataset = dataset
        self.num_classes = 10
        self.resizeW=resizeW
        self.resizeH=resizeH
        self.augment = augment
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),     
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25, hue=0.1),
            v2.RandomAffine(10, (0.05, 0.05), fill=(255, 255, 255))])

        self.norm = v2.Compose([
            v2.Resize((self.resizeH, self.resizeW)),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),                
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = image.convert('RGB')
        if self.augment:
            image = self.transforms(image)
        image = self.norm(image)

        label_one_hot = F.one_hot(torch.tensor(label), self.num_classes).float()
        return image.permute(1, 2, 0), label_one_hot
    
class CRC_Dataset(VisionDataset):
    def __init__(self, paths, resizeW, resizeH, train=True, augment=False, nonorm=False):
        self.num_classes=9
        self.resizeW=resizeW
        self.resizeH=resizeH
        self.augment=augment
        if train:
            if nonorm:
                self.paths = os.path.join(paths, 'NCT-CRC-HE-100K-NONORM')
            else:
                self.paths = os.path.join(paths, 'NCT-CRC-HE-100K')
        else:
            self.paths = os.path.join(paths, 'CRC-VAL-HE-7K')

        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25, hue=0.1),
            v2.RandomAffine(10, (0.05, 0.05), fill=(255, 255, 255)),
            ])
        self.norm = v2.Compose([
            v2.Resize((self.resizeH, self.resizeW)),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],std=[0.21716536, 0.26081574, 0.20723464])])

        self.image_paths = []
        self.labels = []

        self.name_labels = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
        self.class_to_idx = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}        
        for label in os.listdir(self.paths):
            for img_file in os.listdir(os.path.join(self.paths, label)):
                self.image_paths.append(os.path.join(self.paths, label, img_file))
                self.labels.append(self.class_to_idx[label])
        
        show_distribution_all(self.labels, self.name_labels, name="data_distribution_CRC_{}.png".format("train" if train else "val"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])

        if self.augment:
            image = self.transforms(image)
        image=self.norm(image)

        label=torch.zeros(self.num_classes)
        label[self.labels[idx]]=1

        return image.permute(1,2,0), label
    
class PatchCamelyon_Dataset(VisionDataset):
    _FILES = {
        "train": {
            "images": (
                "camelyonpatch_level_2_split_train_x.h5",  # Data file name
                "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",  # Google Drive ID
                "1571f514728f59376b705fc836ff4b63",  # md5 hash
            ),
            "targets": (
                "camelyonpatch_level_2_split_train_y.h5",
                "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
                "35c2d7259d906cfc8143347bb8e05be7",
            ),
        },
        "test": {
            "images": (
                "camelyonpatch_level_2_split_test_x.h5",
                "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
                "d8c2d60d490dbd479f8199bdfa0cf6ec",
            ),
            "targets": (
                "camelyonpatch_level_2_split_test_y.h5",
                "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
                "60a7035772fbdb7f34eb86d4420cf66a",
            ),
        },
        "val": {
            "images": (
                "camelyonpatch_level_2_split_valid_x.h5",
                "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
                "d5b63470df7cfa627aeec8b9dc0c066e",
            ),
            "targets": (
                "camelyonpatch_level_2_split_valid_y.h5",
                "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
                "2b85f58b927af9964a4c15b8f7e8f179",
            ),
        },
    }

    def __init__(self, root, split, resizeW, resizeH, augment = False, download = False):
        super().__init__(root, transform=None, target_transform=None)
        self._split = verify_str_arg(split, "split", ("train", "test", "val"))
        self._base_folder = pathlib.Path(root)
        self.cache_img = OrderedDict()
        self.cache_tgt = OrderedDict()
        self.max_cache_length = 4
        self.resizeW=resizeW
        self.resizeH=resizeH
        self.augment = augment

        self.transforms = v2.Compose([
            v2.RandomRotation([0,360]),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25, hue=0.1),
            v2.RandomAffine(10, (0.05, 0.05), fill=(255, 255, 255)),
            ])
        self.norm = v2.Compose([
            v2.Resize((self.resizeH, self.resizeW)),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.70322989, 0.53606487, 0.66096631],std=[0.21716536, 0.26081574, 0.20723464])])
        #  imagenet (mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # if download:
        #     self._download()
        # if not self._check_exists():
        #     raise RuntimeError("Dataset not found. You can use download=True to download it")

    def __len__(self) -> int:
        images_file = self._FILES[self._split]["images"][0]
        with h5py.File(self._base_folder / images_file, "r") as images_data:
            return images_data["x"].shape[0]

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        images_file = self._FILES[self._split]["images"][0]
        if images_file in self.cache_img:
            image = Image.fromarray(self.cache_img[images_file][idx]).convert("RGB")
        else:
            if len(self.cache_img) > self.max_cache_length:
                self.cache_img.popitem(last=False)
            with h5py.File(self._base_folder / images_file, "r") as images_data:
                self.cache_img[images_file] = images_data["x"][:]
                image = Image.fromarray(self.cache_img[images_file][idx]).convert("RGB")

        targets_file = self._FILES[self._split]["targets"][0]
        if targets_file in self.cache_tgt:
            target = int(self.cache_tgt[targets_file][idx])
        else:
            if len(self.cache_tgt) > self.max_cache_length:  # shape is [num_images, 1, 1, 1]
                self.cache_tgt.popitem(last=False)
            with h5py.File(self._base_folder / targets_file, "r") as targets_data:
                self.cache_tgt[targets_file] = targets_data["y"][:,0,0,0,]
                target = int(self.cache_tgt[targets_file][idx]) 

        if self.augment:
            image = self.transforms(image)
        image = self.norm(image)

        labels = torch.zeros(2)
        labels[target] = 1
        return image.permute(1,2,0), labels
    
    # def _check_exists(self) -> bool:
    #     images_file = self._FILES[self._split]["images"][0]
    #     targets_file = self._FILES[self._split]["targets"][0]
    #     return all(self._base_folder.joinpath(h5_file).exists() for h5_file in (images_file, targets_file))

    # def _download(self) -> None:
    #     if self._check_exists():
    #         print("Files already downloaded and verified")
    #         return
    
    #     for file_name, file_id, md5 in self._FILES[self._split].values():
    #         archive_name = file_name + ".gz"
    #         download_file_from_google_drive(file_id, str(self._base_folder), filename=archive_name, md5=md5)
    #         _decompress(str(self._base_folder / archive_name))

class OCT(data.Dataset):
    def __init__(self, image_paths, labels, resizeW, resizeH, augment=False):
        self.num_classes=12
        self.image_paths=image_paths
        self.labels=labels
        self.resizeW=resizeW
        self.resizeH=resizeH
        self.augment=augment
        self.max_size = 200
        self.transforms = v2.Compose([
            v2.RandomRotation([0,30]),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(10, (0.05, 0.05), fill=0),
            ])
        self.norm = v2.Compose([
            v2.Resize((self.resizeH, self.resizeW), interpolation=1), #1, Lanczos(best), 2, Cubic, 3, Bilinear(fast)
            v2.ToImage(), # (1, H, W)
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.2009],std=[0.1661])
            ])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx]).convert('L') #(W, H)

        if self.augment:
            image = self.transforms(image)
            
        image=self.norm(image) # (1, H, W)
        label=torch.zeros(self.num_classes)
        label[self.labels[idx]]=1

        return image.permute(1,2,0), label # (H, W, 1)