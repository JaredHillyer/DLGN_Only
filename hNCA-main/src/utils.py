import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, balanced_accuracy_score
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import umap
import time
import pickle
import os
from collections import Counter
import torch.nn.functional as F
import psutil

from filelock import FileLock
lock = FileLock("res.pkl.lock")  # Create a lock file associated with res.pkl
cell_labels = ['basophil','eosinophil','erythroblast','myeloblast','promyelocyte','myelocyte','metamyelocyte','neutrophil_banded','neutrophil_segmented','monocyte','lymphocyte_typical','lymphocyte_atypical','smudge_cell']

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def log_vram_usage(text=""):
    max_memory_allocated = torch.cuda.max_memory_allocated() 
    max_memory_reserved = torch.cuda.max_memory_reserved()
    print("\n")
    print(f"Allocated {text}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB ({torch.cuda.memory_allocated() / 1024:.2f} KB)")
    print(f"Reserved {text}: {torch.cuda.memory_reserved() / 1024**2:.2f} MB ({torch.cuda.memory_reserved() / 1024:.2f} KB)")
    print(f"Max allocated: {max_memory_allocated / 1024**2:.2f} MB ({max_memory_allocated / 1024:.2f} KB)")
    print(f"Max reserved: {max_memory_reserved / 1024**2:.2f} MB ({max_memory_reserved / 1024:.2f} KB)")

def showimg(img, path, cmap=None):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    plt.show()
    plt.savefig(path)
    plt.close()

def show_distribution_all(labels, name_labels, name):
    print("Number of samples: ", len(labels))
    y=[Counter(labels)[i] for i in range(len(name_labels))]
    plt.figure(figsize=(10, 5))
    plt.xticks(rotation=45, ha="right")
    plt.bar(name_labels, y)
    plt.rcParams.update({'font.size': 8})
    # plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_loss(train, val, output_path):
    #plots train and validation loss curve
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = [10, 5]
    plt.plot(train, label="Train")
    plt.plot(val, label="Val")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(output_path)
    return

def make_seed(img, channel_n, device):
    # seed = torch.zeros((img.shape[0], img.shape[1], img.shape[2], channel_n), dtype=torch.float32).to(device)
    # seed[..., 0:img.shape[-1]] = img
    seed = F.pad(img, (0, channel_n - img.shape[-1]), mode='constant', value=0)
    return seed

def evaluate_model(test_loader, experiment, model, num_classes):
    predictions = []
    labels = []
    with torch.no_grad():
        for image, label in test_loader:
            #for cnn baselines: mobilenet_v2, resnet18, if experiment name contains "mobilenet_v2" or "resnet18"
            if "net" in experiment:
                image = image.permute(0,3,1,2).to("cuda")
            out = model(image)#[0]
            pred = np.argmax(out.detach().cpu().numpy(), axis=1)  
            label = np.argmax(label.cpu().numpy(), axis=1)        
            predictions.extend(pred)  
            labels.extend(label)    

    accuracy = accuracy_score(labels, predictions)
    b_accuracy = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    print(f"{experiment} accuracy: {accuracy:.4f}")
    print(f"{experiment} balanced accuracy: {b_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nClassification Report:\n", classification_report(labels, predictions, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(labels, predictions))

    res = {}
    with lock:
        if os.path.exists("res.pkl"):
            try:
                with open("res.pkl", "rb") as f:
                    res = pickle.load(f)
            except EOFError:
                print("Warning: res.pkl is empty or corrupted. Initializing as empty dictionary.")
                res = {}

        res[f"{experiment}"] = {"accuracy": accuracy, \
            "b_accuracy": b_accuracy,\
            "f1": f1,\
            "classification_report": classification_report(labels, predictions, zero_division=0),\
            "confusion_matrix": confusion_matrix(labels, predictions)}
        
        with open(f"res.pkl", "wb") as f:
            pickle.dump(res, f)