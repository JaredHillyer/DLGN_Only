import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold
import data.Dataset as Dataset 
import numpy as np
from collections import Counter
import torch
import torchvision
from utils import show_distribution_all
from data.GetData import *
from data.DataInfo import *
from collections import defaultdict
import random

def load_datasets(train_set, fold, resizeW, resizeH, batch_size, criterion):
    if criterion == "Focal":
        balance = False
    elif criterion == "CE":
        balance = True

    if train_set=="AML":
        train_loader, val_loader = load_dataset_AML(fold, resizeW, resizeH, batch_size, balance)
    elif train_set=="PBC":
        train_loader, val_loader = load_dataset_PBC(fold, resizeW, resizeH, batch_size, balance)
    elif train_set=="MLL":
        train_loader, val_loader = load_dataset_MLL(fold, resizeW, resizeH, batch_size, balance)
    elif train_set=="Malaria":
        train_loader, val_loader = load_dataset_Malaria(fold, resizeW, resizeH, batch_size, balance)
    elif train_set=="SIPAKMED":
        train_loader, val_loader = load_dataset_SIPAKMED(fold, resizeW, resizeH, batch_size, balance)
    elif train_set=="Urine":
        train_loader, val_loader = load_dataset_Urine(fold, resizeW, resizeH, batch_size, balance)
    elif train_set=="CIFAR10":
        train_loader, val_loader = load_dataset_CIFAR10(resizeW, resizeH, batch_size)
    elif train_set=="CRC":
        train_loader, val_loader = load_dataset_CRC(resizeW, resizeH, batch_size)
    elif train_set=="CRC_nonorm":
        train_loader, val_loader = load_dataset_CRC(resizeW, resizeH, batch_size, nonorm=True)
    elif train_set=="PatchCamelyon":
        train_loader, val_loader, test_loader = load_dataset_PatchCamelyon(resizeW, resizeH, batch_size)
    elif train_set=="OCT":
        train_loader, val_loader = load_dataset_OCT(fold, resizeW, resizeH, batch_size, balance)

    return train_loader, val_loader

def get_weights(y):
    class_counts = Counter(y)
    class_counts = np.array([class_counts[i] for i in range(15)])
    class_weights = 1 / (class_counts + 0.001)
    sample_weights = [class_weights[i] for i in y]
    return sample_weights

# def load_dataset_AML(fold, resizeW, resizeH, batch_size, balance=False):
#     X_AML, y_AML = get_data_AML(AML_data_path)
#     X_AML = np.asarray(X_AML)
#     y_AML = np.asarray(y_AML)

#     # --- Limit each class to a maximum of n samples ---
#     n = 100  # Maximum number of samples per class
#     class_indices = defaultdict(list)
#     for idx, label in enumerate(y_AML):
#         class_indices[label].append(idx)

#     selected_indices = []
#     for label, indices in class_indices.items():
#         if len(indices) > n:
#             selected_indices.extend(random.sample(indices, n))
#         else:
#             selected_indices.extend(indices)

#     # Shuffle selected indices
#     random.shuffle(selected_indices)

#     # Subset the data
#     X_AML = X_AML[selected_indices]
#     y_AML = y_AML[selected_indices]

#     # Split into train/test (80/20 split)
#     split_idx = int(0.8 * len(X_AML))
#     X_AML_train, X_AML_test = X_AML[:split_idx], X_AML[split_idx:]
#     y_AML_train, y_AML_test = y_AML[:split_idx], y_AML[split_idx:]

#     # Create datasets
#     AML_train_dataset = Dataset.WBC_Dataset(X_AML_train, y_AML_train, augment=True, resizeW=resizeW, resizeH=resizeH, dataset="AML")
#     AML_val_dataset = Dataset.WBC_Dataset(X_AML_test, y_AML_test, resizeW=resizeW, resizeH=resizeH, dataset="AML")

#     # Create samplers
#     if balance:
#         AML_sampler = data.WeightedRandomSampler(weights=get_weights(y_AML_train), num_samples=len(AML_train_dataset), replacement=True)
#     else:
#         AML_sampler = data.RandomSampler(AML_train_dataset)

#     # Create loaders
#     AML_train_loader = data.DataLoader(AML_train_dataset, sampler=AML_sampler, batch_size=batch_size)
#     AML_val_loader = data.DataLoader(AML_val_dataset, batch_size=batch_size)

#     return AML_train_loader, AML_val_loader

def load_dataset_AML(fold, resizeW, resizeH, batch_size, balance=False):
    
    fold = fold-1

    X_AML,y_AML = get_data_AML(AML_data_path)
    X_AML=np.asarray(X_AML)
    y_AML=np.asarray(y_AML)

    skf_AML = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf_AML.get_n_splits(X_AML, y_AML)
    for i,(train_index, test_index) in enumerate(skf_AML.split(X_AML, y_AML)):
        if i != fold:
            continue
        X_AML_train, X_AML_test = X_AML[train_index], X_AML[test_index]
        y_AML_train, y_AML_test = y_AML[train_index], y_AML[test_index]

    AML_train_dataset = Dataset.WBC_Dataset(X_AML_train,y_AML_train, augment=True, resizeW=resizeW, resizeH=resizeH, dataset="AML")
    AML_val_dataset = Dataset.WBC_Dataset(X_AML_test,y_AML_test, resizeW=resizeW, resizeH=resizeH,dataset="AML")

    if balance:
        AML_sampler = data.WeightedRandomSampler(weights=get_weights(y_AML_train), num_samples=len(AML_train_dataset), replacement=True)
    else:
        AML_sampler = data.RandomSampler(AML_train_dataset)

    AML_train_loader = data.DataLoader(AML_train_dataset,sampler=AML_sampler,batch_size=batch_size)
    AML_val_loader = data.DataLoader(AML_val_dataset, batch_size=batch_size)
   
    return AML_train_loader, AML_val_loader

def load_dataset_PBC(fold, resizeW, resizeH, batch_size, balance=False):
    
    fold = fold-1

    X_PBC,y_PBC = get_data_PBC(PBC_data_path)
    X_PBC=np.asarray(X_PBC)
    y_PBC=np.asarray(y_PBC)

    skf_PBC = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf_PBC.get_n_splits(X_PBC, y_PBC)
    for i,(train_index, test_index) in enumerate(skf_PBC.split(X_PBC, y_PBC)):
        if i != fold:
            continue
        X_PBC_train, X_PBC_test = X_PBC[train_index], X_PBC[test_index]
        y_PBC_train, y_PBC_test = y_PBC[train_index], y_PBC[test_index]

    PBC_train_dataset = Dataset.WBC_Dataset(X_PBC_train,y_PBC_train, augment=True, resizeW=resizeW, resizeH=resizeH, dataset="PBC")
    PBC_val_dataset = Dataset.WBC_Dataset(X_PBC_test,y_PBC_test, resizeW=resizeW, resizeH=resizeH, dataset="PBC")

    if balance:
        PBC_sampler = data.WeightedRandomSampler(weights=get_weights(y_PBC_train), num_samples=len(PBC_train_dataset), replacement=True)
    else:
        PBC_sampler = data.RandomSampler(PBC_train_dataset)

    PBC_train_loader = data.DataLoader(PBC_train_dataset,sampler=PBC_sampler,batch_size=batch_size)
    PBC_val_loader = data.DataLoader(PBC_val_dataset, batch_size=batch_size)

    return PBC_train_loader, PBC_val_loader

def load_dataset_MLL(fold, resizeW, resizeH, batch_size, balance=False):
    
    fold = fold-1

    X_MLL,y_MLL = get_data_MLL(MLL_data_path)
    X_MLL=np.asarray(X_MLL)
    y_MLL=np.asarray(y_MLL)

    skf_MLL = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf_MLL.get_n_splits(X_MLL, y_MLL)
    for i,(train_index, test_index) in enumerate(skf_MLL.split(X_MLL, y_MLL)):
        if i != fold:
            continue
        X_MLL_train, X_MLL_test = X_MLL[train_index], X_MLL[test_index]
        y_MLL_train, y_MLL_test = y_MLL[train_index], y_MLL[test_index]

    MLL_train_dataset = Dataset.WBC_Dataset(X_MLL_train,y_MLL_train, augment=True, resizeW=resizeW, resizeH=resizeH, dataset="MLL")
    MLL_val_dataset = Dataset.WBC_Dataset(X_MLL_test,y_MLL_test, resizeW=resizeW, resizeH=resizeH, dataset="MLL")

    if balance:
        MLL_sampler = data.WeightedRandomSampler(weights=get_weights(y_MLL_train), num_samples=len(MLL_train_dataset), replacement=True)
    else:
        MLL_sampler = data.RandomSampler(MLL_train_dataset)

    MLL_train_loader = data.DataLoader(MLL_train_dataset, sampler = MLL_sampler, batch_size = batch_size)
    MLL_val_loader = data.DataLoader(MLL_val_dataset, batch_size=batch_size)

    return MLL_train_loader, MLL_val_loader

def load_dataset_Malaria(fold, resizeW, resizeH, batch_size, balance=False):
    
    fold = fold-1

    X_Malaria,y_Malaria = get_data_Malaria(Malaria_data_path)
    X_Malaria=np.asarray(X_Malaria)
    y_Malaria=np.asarray(y_Malaria)

    skf_Malaria = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf_Malaria.get_n_splits(X_Malaria, y_Malaria)
    for i,(train_index, test_index) in enumerate(skf_Malaria.split(X_Malaria, y_Malaria)):
        if i != fold:
            continue
        X_Malaria_train, X_Malaria_test = X_Malaria[train_index], X_Malaria[test_index]
        y_Malaria_train, y_Malaria_test = y_Malaria[train_index], y_Malaria[test_index]

    Malaria_train_dataset = Dataset.Malaria(X_Malaria_train,y_Malaria_train, augment=True, resizeW=resizeW, resizeH=resizeH)
    Malaria_val_dataset = Dataset.Malaria(X_Malaria_test,y_Malaria_test, resizeW=resizeW, resizeH=resizeH)

    if balance:
        Malaria_sampler = data.WeightedRandomSampler(weights=get_weights(y_Malaria_train), num_samples=len(Malaria_train_dataset), replacement=True)
    else:
        Malaria_sampler = data.RandomSampler(Malaria_train_dataset)

    Malaria_train_loader = data.DataLoader(Malaria_train_dataset,sampler=Malaria_sampler,batch_size=batch_size)
    Malaria_val_loader = data.DataLoader(Malaria_val_dataset, batch_size=batch_size)

    return Malaria_train_loader, Malaria_val_loader

def load_dataset_SIPAKMED(fold, resizeW, resizeH, batch_size, balance=False):
    
    fold = fold-1

    X_SIPAKMED,y_SIPAKMED = get_data_SIPAKMED(SIPAKMED_data_path)
    X_SIPAKMED=np.asarray(X_SIPAKMED)
    y_SIPAKMED=np.asarray(y_SIPAKMED)

    skf_SIPAKMED = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf_SIPAKMED.get_n_splits(X_SIPAKMED, y_SIPAKMED)
    for i,(train_index, test_index) in enumerate(skf_SIPAKMED.split(X_SIPAKMED, y_SIPAKMED)):
        if i != fold:
            continue
        X_SIPAKMED_train, X_SIPAKMED_test = X_SIPAKMED[train_index], X_SIPAKMED[test_index]
        y_SIPAKMED_train, y_SIPAKMED_test = y_SIPAKMED[train_index], y_SIPAKMED[test_index]

    SIPAKMED_train_dataset = Dataset.SIPAKMED(X_SIPAKMED_train,y_SIPAKMED_train, augment=True, resizeW=resizeW, resizeH=resizeH)
    SIPAKMED_val_dataset = Dataset.SIPAKMED(X_SIPAKMED_test,y_SIPAKMED_test, resizeW=resizeW, resizeH=resizeH)

    if balance:
        SIPAKMED_sampler = data.WeightedRandomSampler(weights=get_weights(y_SIPAKMED_train), num_samples=len(SIPAKMED_train_dataset), replacement=True)
    else:
        SIPAKMED_sampler = data.RandomSampler(SIPAKMED_train_dataset)

    SIPAKMED_train_loader = data.DataLoader(SIPAKMED_train_dataset,sampler=SIPAKMED_sampler,batch_size=batch_size)
    SIPAKMED_val_loader = data.DataLoader(SIPAKMED_val_dataset, batch_size=batch_size)

    return SIPAKMED_train_loader, SIPAKMED_val_loader

def load_dataset_Urine(fold, resizeW, resizeH, batch_size, balance=False):
    
    fold = fold-1

    X_Urine,y_Urine = get_data_Urine(Urine_data_path)
    X_Urine=np.asarray(X_Urine)
    y_Urine=np.asarray(y_Urine)

    skf_Urine = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf_Urine.get_n_splits(X_Urine, y_Urine)
    for i,(train_index, test_index) in enumerate(skf_Urine.split(X_Urine, y_Urine)):
        if i != fold:
            continue
        X_Urine_train, X_Urine_test = X_Urine[train_index], X_Urine[test_index]
        y_Urine_train, y_Urine_test = y_Urine[train_index], y_Urine[test_index]

    Urine_train_dataset = Dataset.Urine(X_Urine_train,y_Urine_train, augment=True, resizeW=resizeW, resizeH=resizeH)
    Urine_val_dataset = Dataset.Urine(X_Urine_test,y_Urine_test, resizeW=resizeW, resizeH=resizeH)

    if balance:
        Urine_sampler = data.WeightedRandomSampler(weights=get_weights(y_Urine_train), num_samples=len(Urine_train_dataset), replacement=True)
    else:
        Urine_sampler = data.RandomSampler(Urine_train_dataset)

    Urine_train_loader = data.DataLoader(Urine_train_dataset,sampler=Urine_sampler,batch_size=batch_size)
    Urine_val_loader = data.DataLoader(Urine_val_dataset, batch_size=batch_size)

    return Urine_train_loader, Urine_val_loader

def load_dataset_CIFAR10(resizeW, resizeH, batch_size):
    train_dataset = torchvision.datasets.CIFAR10(root=CIFAR10_data_path, train=True, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root=CIFAR10_data_path, train=False, download=True)
    
    train_dataset_onehot = Dataset.CIFAR10WithOneHot(train_dataset, resizeW=resizeW, resizeH=resizeH, augment=True)
    val_dataset_onehot = Dataset.CIFAR10WithOneHot(val_dataset, resizeW=resizeW, resizeH=resizeH)

    train_loader = data.DataLoader(train_dataset_onehot, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset_onehot, batch_size=batch_size, shuffle=False)

    name_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    y = [i[1] for i in train_dataset]
    show_distribution_all(y, name_labels, name="data_distribution_CIFAR10_train.png")
    y = [i[1] for i in val_dataset]
    show_distribution_all(y, name_labels, name="data_distribution_CIFAR10_val.png")

    return train_loader, val_loader

def load_dataset_CRC(resizeW, resizeH, batch_size, nonorm=False):
    train_dataset = Dataset.CRC_Dataset(CRC_data_path, resizeW=resizeW, resizeH=resizeH, train=True, augment=True, nonorm=nonorm)
    val_dataset = Dataset.CRC_Dataset(CRC_data_path, resizeW=resizeW, resizeH=resizeH, train=False)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def load_dataset_PatchCamelyon(resizeW, resizeH, batch_size):

    train_dataset = Dataset.PatchCamelyon_Dataset(PatchCamelyon_data_path, split="train", resizeW=resizeW, resizeH=resizeH, augment=True)
    val_dataset = Dataset.PatchCamelyon_Dataset(PatchCamelyon_data_path, split="val", resizeW=resizeW, resizeH=resizeH)
    test_dataset = Dataset.PatchCamelyon_Dataset(PatchCamelyon_data_path, split="test", resizeW=resizeW, resizeH=resizeH)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # for i, (image, target) in enumerate(train_loader):
    #     print(image.shape)
    #     print(target)
    #     break

    name_labels = ['normal', 'tumor']
    y = [i[1] for i in train_dataset]
    # transform the tensor one-hot labels y to integer labels
    y = [torch.argmax(i).item() for i in y]
    show_distribution_all(y, name_labels, name="data_distribution_PatchCamelyon_train.png")
    y = [i[1] for i in val_dataset]
    y = [torch.argmax(i).item() for i in y]
    show_distribution_all(y, name_labels, name="data_distribution_PatchCamelyon_val.png")
    y = [i[1] for i in test_dataset]
    y = [torch.argmax(i).item() for i in y]
    show_distribution_all(y, name_labels, name="data_distribution_PatchCamelyon_test.png")

    return train_loader, val_loader, test_loader

def load_dataset_OCT(fold, resizeW, resizeH, batch_size, balance=False):
    
    fold = fold-1

    X_OCT,y_OCT = get_data_OCT(OCT_data_path)

    X_OCT=np.asarray(X_OCT)
    y_OCT=np.asarray(y_OCT)

    skf_OCT = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    skf_OCT.get_n_splits(X_OCT, y_OCT)
    for i,(train_index, test_index) in enumerate(skf_OCT.split(X_OCT, y_OCT)):
        if i != fold:
            continue
        X_OCT_train, X_OCT_test = X_OCT[train_index], X_OCT[test_index]
        y_OCT_train, y_OCT_test = y_OCT[train_index], y_OCT[test_index]

    OCT_train_dataset = Dataset.OCT(X_OCT_train,y_OCT_train, augment=True, resizeW=resizeW, resizeH=resizeH)
    OCT_val_dataset = Dataset.OCT(X_OCT_test,y_OCT_test, resizeW=resizeW, resizeH=resizeH)

    # # Set augment=False
    # data = torch.stack([OCT_train_dataset[i][0] for i in range(len(OCT_train_dataset))])  # (N, H, W, C)
    # mean = data.mean(dim=(0, 1, 2))
    # std = data.std(dim=(0, 1, 2))
    # print(f"Mean: {mean}")
    # print(f"Std: {std}")

    if balance:
        OCT_sampler = data.WeightedRandomSampler(weights=get_weights(y_OCT_train), num_samples=len(OCT_train_dataset), replacement=True)
    else:
        OCT_sampler = data.RandomSampler(OCT_train_dataset)

    OCT_train_loader = data.DataLoader(OCT_train_dataset,sampler=OCT_sampler,batch_size=batch_size)
    OCT_val_loader = data.DataLoader(OCT_val_dataset, batch_size=batch_size)

    return OCT_train_loader, OCT_val_loader
