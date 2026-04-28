import data.Dataset as Dataset 
import os
from torchvision.transforms import v2
from PIL import Image
import torch
import torchvision
from utils import show_distribution_all, cell_labels
from data.DataInfo import *

def get_data_AML(data_path):
    image_paths = []
    labels = []

    for dirs in os.listdir(data_path):
        folder_path = os.path.join(data_path, dirs)
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.tiff'):
                image_path = os.path.join(folder_path, file)
                
                if "BAS" in file:
                    label = 0
                elif "EBO" in file:
                    label = 2
                elif "EOS" in file:
                    label = 1
                elif "KSC" in file:
                    label = 12
                elif "LYA" in file:
                    label = 11
                elif "LYT" in file:
                    label = 10
                elif "MMZ" in file:
                    label = 6
                elif "MOB" in file:
                    label = 9
                elif "MON" in file:
                    label = 9
                elif "MYB" in file:
                    label = 5
                elif "MYO" in file:
                    label = 3
                elif "NGB" in file:
                    label = 7
                elif "NGS" in file:
                    label = 8
                elif "PMB" in file:
                    continue
                    label = 13
                elif "PMO" in file:
                    label = 4
                labels.append(label)
                image_paths.append(image_path)
    
    show_distribution_all(labels, cell_labels, name="data_distribution_AML.png")

    return image_paths, labels

def get_data_PBC(data_path):
    image_paths = []
    labels = []

    for dirs in os.listdir(data_path):
        if ".DS_" in dirs:
            continue
        folder_path = os.path.join(data_path, dirs)
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.tiff'):
                image_path = os.path.join(folder_path, file)
                    
                if "basophil" in dirs:
                    label = 0
                elif "eosinophil" in dirs:
                    label = 1
                elif "erythroblast" in dirs:
                    label = 2
                elif "IG" in dirs:
                    continue
                    label = 13
                elif "PMY" in dirs:
                    label = 4
                elif "MY" in dirs:
                    label = 5
                    if "MMY" in dirs:
                        label = 6
                elif "lymphocyte" in dirs:
                    label = 10
                elif "monocyte" in dirs:
                    label = 9
                elif "NEUTROPHIL" in dirs:
                    continue
                    label = 13
                elif "BNE" in file:
                    label = 7
                elif "SNE" in file:
                    label = 8
                elif "platelet" in file:
                    continue
                    label = 13
                labels.append(label)
                image_paths.append(image_path)
        
    show_distribution_all(labels, cell_labels, name="data_distribution_PBC.png")

    return image_paths, labels

def get_data_MLL(data_path):
    image_paths = []
    labels = []

    for dirs in os.listdir(data_path):
        folder_path = os.path.join(data_path, dirs)
       
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.tiff') or file.endswith('.TIF'):
                image_path = os.path.join(folder_path, file)
                    
                if "01" in dirs:
                    label = 2
                elif "04" in dirs:
                    continue
                    label = 13
                elif "05" in dirs:
                    label = 9
                elif "08" in dirs:
                    label = 11
                elif "09" in dirs:
                    label = 0
                elif "10" in dirs:
                    label = 1
                elif "11" in dirs:
                    label = 7
                elif "12" in dirs:
                    label = 11
                elif "13" in dirs:
                    label = 3
                elif "14" in dirs:
                    label = 10
                elif "15" in dirs:
                    label = 8
                elif "16" in dirs:
                    continue
                    label = 13
                elif "17" in dirs:
                    label = 12
                elif "18" in dirs:
                    label = 4
                elif "19" in dirs:
                    label = 5
                elif "20" in dirs:
                    label = 6
                elif "21" in dirs:
                    continue
                    label = 13
                elif "22" in dirs:
                    continue
                    label = 13
                labels.append(label)
                image_paths.append(image_path)
    
    show_distribution_all(labels, cell_labels, name="data_distribution_MLL.png")

    return image_paths, labels

def get_data_Malaria(data_path):
    image_paths = []
    labels = []
    i = 0

    for dirs in os.listdir(data_path):
        folder_path = os.path.join(data_path, dirs)
        for file in os.listdir(folder_path):
            if file.endswith(".png"):
                image_path = os.path.join(folder_path, file)
                image_paths.append(image_path)
                labels.append(i)
        i+=1
    name_labels = ['Parasitized', 'Uninfected']
    show_distribution_all(labels, name_labels, name="data_distribution_Malaria.png")

    return image_paths, labels

def get_data_SIPAKMED(data_path):
    image_paths = []
    labels = []
    i = 0

    for dirs in os.listdir(data_path):
        folder_path = os.path.join(data_path, dirs, dirs, "CROPPED")
        for file in os.listdir(folder_path):
            if file.endswith(".bmp"):
                image_path = os.path.join(folder_path, file)
                image_paths.append(image_path)
                labels.append(i)
        i+=1

    name_labels = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
    show_distribution_all(labels, name_labels, name="data_distribution_SIPAKMED.png")

    return image_paths, labels

def get_data_Urine(data_path):
    image_paths = []
    labels = []
    i = 0

    for dirs in os.listdir(data_path):
        folder_path = os.path.join(data_path, dirs, dirs)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".jpg"):
                image_path = os.path.join(folder_path, file)
                image_paths.append(image_path)
                labels.append(i)
        i+=1
    
    name_labels = ['bacteria', 'crystal', 'cylinder', 'epithelial', 'erythrocyte', 'leukocyte', 'others', 'yeast']
    show_distribution_all(labels, name_labels, name="data_distribution_Urine.png")

    return image_paths, labels

def get_data_OCT(data_path):

    image_paths = []
    labels = []
    
    name_labels = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'ERM', 'MH', 'NO', 'RAO', 'RVO', 'VID']
    class_to_idx = {'AMD': 0, 'CNV': 1, 'CSR': 2, 'DME': 3, 'DR': 4, 'DRUSEN': 5, 'ERM': 6, 'MH': 7, 'NO': 8, 'RAO': 9, 'RVO': 10, 'VID': 11}        
    for dir in os.listdir(data_path):
        folder_path = os.path.join(data_path, dir)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(folder_path, file))
                labels.append(class_to_idx[dir])
    show_distribution_all(labels, name_labels, name="data_distribution_OCT.png")

    return image_paths, labels


def get_test_sample(resizeW, resizeH, index=0, dataset="AML"):
    print(f"Get test sample {index} from {dataset}")
    if dataset=="AML":
        norm_aml = v2.Compose([v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.82069695, 0.7281261, 0.836143],std=[0.16157213, 0.2490039, 0.09052657])])
        x,y = get_data_AML(AML_data_path)
        img = Image.open(x[index]).convert('RGB')
        img = img.resize((resizeH, resizeW))
        img = norm_aml(img).permute(1,2,0)
        y = y[index]
    elif dataset=="PBC":
        norm_pbc = v2.Compose([v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.8746204, 0.7487587, 0.7203138],std=[0.15061052, 0.17617777, 0.07467376])])
        x,y = get_data_PBC(PBC_data_path)
        img = Image.open(x[index]).convert('RGB')
        img = img.resize((resizeH, resizeW))
        img = norm_pbc(img).permute(1,2,0)
        y = y[index]
    elif dataset=="MLL":
        norm_mll = v2.Compose([v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.74053776, 0.6514114, 0.7785342],std=[0.18301032, 0.24672535, 0.16100405])])  
        x,y = get_data_MLL(MLL_data_path)
        img = Image.open(x[index]).convert('RGB')
        img = img.resize((resizeH, resizeW))
        img = norm_mll(img).permute(1,2,0)
        y = y[index]
    elif dataset=="SIPAKMED":
        norm_sipakmed = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.5758177042007446, 0.5742955207824707, 0.6884000301361084],std=[0.2185823619365692, 0.19480502605438232, 0.19809827208518982])])
        x,y = get_data_SIPAKMED(SIPAKMED_data_path)
        img = Image.open(x[index]).convert('RGB')
        img = img.resize((resizeH, resizeW))
        img = norm_sipakmed(img).permute(1,2,0)
        y = y[index]
    elif dataset=="Urine":
        norm_urine = v2.Compose([
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.6369247436523438, 0.7650187611579895, 0.7165603637695312],std=[0.05541696399450302, 0.05346846953034401, 0.056218646466732025])])
        x,y = get_data_Urine(Urine_data_path)
        img = Image.open(x[index]).convert('RGB')
        img = img.resize((resizeH, resizeW))
        img = norm_urine(img).permute(1,2,0)
        y = y[index]
    elif dataset=="Malaria":
        norm_malaria = v2.Compose([v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.5286651253700256, 0.42176899313926697, 0.4479588568210602],std=[0.34576717019081116, 0.2789415419101715, 0.29265928268432617])])
        x,y = get_data_Malaria(Malaria_data_path)
        img = Image.open(x[index]).convert('RGB')
        img = img.resize((resizeH, resizeW))
        img = norm_malaria(img).permute(1,2,0)
        y = y[index]
    elif dataset=="CIFAR10":
        norm_cifar10 = v2.Compose([v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),                
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
        train_dataset = torchvision.datasets.CIFAR10(root=CIFAR10_data_path, train=True, download=True)
        x,y = train_dataset[index]
        img = x.resize((resizeH, resizeW))
        img = norm_cifar10(img).permute(1,2,0)
    elif dataset=="CRC":
        img, y = Dataset.CRC_Dataset(CRC_data_path, resizeW=resizeW, resizeH=resizeH, train=False)[index]
        y = torch.argmax(y).item()
    elif dataset=="PatchCamelyon":
        img, y = Dataset.PatchCamelyon_Dataset(PatchCamelyon_data_path, split="val", resizeW=resizeW, resizeH=resizeH)[index]
        y = torch.argmax(y).item()
    elif dataset=="OCT":
        norm_oct = v2.Compose([v2.Resize((resizeH, resizeW), interpolation=1), # (H, W, 1)
            v2.ToImage(), # (1, H, W)
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=[0.2009],std=[0.1661])])
        x,y = get_data_OCT(OCT_data_path)
        img = Image.open(x[index]).convert('L') # (W, H)
        img = norm_oct(img).permute(1,2,0) # (H, W, 1)
        y = y[index]
    return img, y