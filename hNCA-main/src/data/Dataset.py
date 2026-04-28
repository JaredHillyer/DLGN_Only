import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import v2

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