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

def showsubplots(imgs, path, cmap=None):
    # fig, axs = plt.subplots(8, 16, figsize=(64, 32))
    # for i in range(8):
    #     for j in range(16):
    #         axs[i, j].imshow(imgs[i*8+j], cmap=cmap)
    #         axs[i, j].axis('off')
    # plt.colorbar()
    # plt.show()
    # plt.savefig(path)
    # plt.close()

    num_channels = imgs.shape[-1]
    fig, axes = plt.subplots(8, 16, figsize=(16*2, 8*2))
    axes = axes.flatten()
    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(imgs[:, :, i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'C {i + 1}')
    for j in range(num_channels, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
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

class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr
    
def get_dice_scores(model,agent,test_loader,dataset,steps):
    #computes dice scores for each image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dice_scores=[]
    iterable = iter(test_loader)
    j=-1
    for i in enumerate(test_loader):
        j=j+1
        x,s=next(iterable)
        out,_=model(agent.make_seed(x), steps=steps,fire_rate=0.5)
        out=out.detach()
        input=out
        target=s
        input = torch.sigmoid(input)  
        input = torch.flatten(input)
        target = torch.flatten(target).to(device)
        intersection = (input * target).sum()
        dice = (2.*intersection)/(input.sum() + target.sum())
        dice_scores.append(dice.cpu().item())
        
    print(dice_scores)
    np.savetxt("dice_scores_"+dataset,dice_scores,fmt="%d",delimiter=',')
    print(dice_scores.mean())

def animate_activation(model,agent,val_loader,steps):
    #visualizes channel activations over time
    import matplotlib.animation as animation
    from IPython.display import HTML
    names=["BAS","EBO","EOS","KSC","LYA","LYT","MMZ","MOB","MON","MYB","MYO","NGB","NGS","PMB","PMO"]
    x,s=next(iter(val_loader))
    plt.rcParams["figure.figsize"] = (64,64)
    plt.figure(figsize=(64,64))
    fig, ax = plt.subplots()
    ims = []
    for i in range(steps+1):
        pred,feat_map=model(agent.make_seed(x), steps=i-1,fire_rate=0.5)
        pred=pred.detach()
        feat_map=feat_map.detach()
        sig=torch.nn.Sigmoid()
        feat_map=sig(feat_map)
        feat_map=feat_map[0].cpu()
        feat_map[:,:,0]=0*feat_map[:,:,0]
        feat_map[:,:,1]=0*feat_map[:,:,0]
        feat_map[:,:,2]=0*feat_map[:,:,0]
        #feat_map = torch.reshape(feat_map.permute(0,2,1),(feat_map.shape[0]*8,feat_map.shape[1]*8))#feat_map.shape[2]))
        feat_map=torch.cat([torch.cat([feat_map[:,:,i+8*j] for i in range(8)],axis=1) for j in range(8)],axis=0)
        #feat_map=feat_map.numpy()
        plt.gray()
        im = ax.imshow(feat_map, animated=True)
        if i == 0:
            ax.imshow(feat_map)  # show an initial one first
        ims.append([im])


    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat_delay=10000)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("channel_visualization.gif", writer=writer)
    return HTML(ani.to_jshtml())

def visualize_activations(model,agent,steps,sample,name):
    #plots channels of the NCA
    sample=sample[None,:,:,:]

    plt.figure("visualise", (64,64))
    out,feature_map=model(agent.make_seed(sample), steps=steps,fire_rate=0.5)
    feature_map=feature_map.detach()
    #sig=torch.nn.Sigmoid()
    #feature_map=sig(feature_map)
    feature_map=feature_map.cpu()
    feature_map[0,:,:,0]=0*feature_map[0,:,:,0]
    feature_map[0,:,:,1]=0*feature_map[0,:,:,0]
    feature_map[0,:,:,2]=0*feature_map[0,:,:,0]
    for i in range(8):
        for j in range(8):
            plt.subplot(8,8,8*i+j+1)
            plt.gray()
            plt.imshow(feature_map[0,:,:,8*i+j])
    plt.savefig("channel_activation"+name+".png")
    return

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

def plot_spectrum(img, magnitude_spectrum, phase_spectrum, i, output_dir):
    plt.figure(figsize=(15, 5))
    # Image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image')
    # Magnitude Spectrum
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(magnitude_spectrum + 1e-10), cmap='gray')
    plt.title('Magnitude Spectrum (Log Scale)')
    # Phase Spectrum
    plt.subplot(1, 3, 3)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.title('Phase Spectrum')
    plt.axis('off')

    plt.savefig(output_dir + str(i) +"fft.png")
    plt.show()
    plt.close()

def plot64(img, title, normalize=True, output_dir="figs/"):
    rows = 8
    cols = 8
    fig, axes = plt.subplots(rows, cols, figsize=(100, 100)) 
    for i in range(64):
        ax = axes[i // cols, i % cols] 
        if normalize:
            normalized_img = (img[0, i] - img[0, i].min()) / (img[0, i].max() - img[0, i].min())
            ax.imshow(normalized_img.clone().detach().cpu().numpy(), cmap='gray')
        else:
            ax.imshow(img[0, i].clone().detach().cpu().numpy(), cmap='gray')
        ax.axis('off') 
    plt.tight_layout()
    plt.savefig(output_dir + title)
    plt.show()
    plt.close()

def apply_pass_filter(fre, ratio=1.3,type="low"):
    b, d, h, w = fre.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    center_y, center_x = h // 2, w // 2
    distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)    
    cutoff = torch.mean(distance) * ratio
    if type == "low":
        filter_mask = (distance <= cutoff).float().to(fre.device)
    elif type == "high":
        filter_mask = (distance > cutoff).float().to(fre.device)
    filter_mask = filter_mask.unsqueeze(0).unsqueeze(0)

    filtered_fre = fre * filter_mask
    return filtered_fre


def visualize_tsne_umap(single_labels, umap_path, tsne_path, features_reduced):
    color_map = plt.colormaps['tab20'] 
    num_classes = 13
    colors = [color_map(i) for i in range(num_classes)]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(num_classes + 1)  
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ############################################ UMAP Visualization
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, metric='euclidean', random_state=42, n_jobs=1)
    features_umap = umap_model.fit_transform(features_reduced.detach().cpu().numpy())  # Shape: [total_samples, 2]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], 
                        c=single_labels.numpy(), 
                        cmap=cmap, norm=norm, 
                        alpha=0.7, s=20)  

    cbar = plt.colorbar(scatter, ticks=np.arange(num_classes), boundaries=bounds)
    cbar.set_label("Class Label")
    cbar.set_ticks(np.arange(num_classes) + 0.5)
    cbar.set_ticklabels(cell_labels)

    plt.title("UMAP Projection of Feature Maps with Class Labels")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid()
    plt.savefig(umap_path) 
    plt.show()
    plt.close

    ############################################ t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(features_reduced.detach().cpu().numpy())  

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                        c=single_labels.numpy(), 
                        cmap=cmap, norm=norm, 
                        alpha=0.7, s=20)  

    cbar = plt.colorbar(scatter, ticks=np.arange(num_classes), boundaries=bounds)
    cbar.set_label("Class Label")
    cbar.set_ticks(np.arange(num_classes) + 0.5)
    cbar.set_ticklabels(cell_labels)

    plt.title("t-SNE Visualization of Feature Maps")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid()
    plt.savefig(tsne_path) 
    plt.show()
    plt.close

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
            if "pool1" in experiment or "NCA" in experiment:
                image = image.to("cuda")
            else:
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

    # labels = [cell_labels[i] for i in labels]
    # predictions = [cell_labels[i] for i in predictions]
    # exclude_classes = ["myeloblast", "lymphocyte_atypical", "smudge_cell"]
    # exclude_classes = [cell_labels.index(i) for i in exclude_classes]

    # filtered_labels = []
    # filtered_predictions = []
    # for lbl, pred in zip(labels, predictions):
    #     if lbl not in exclude_classes and pred not in exclude_classes:
    #         filtered_labels.append(lbl)
    #         filtered_predictions.append(pred)

