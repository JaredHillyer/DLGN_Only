#IMPORT
import yaml
import os
import torch
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import KFold
import src.Datasets as Datasets
import src.Models as Models
import src.Losses as Losses
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt

#FUNCTIONS
def apply_pca_segmentation(feature_maps, n_components=1):
    C, H, W = feature_maps.shape 
    X = feature_maps.reshape(C, -1).T 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    pca_images = X_pca.T.reshape(n_components, H, W)
    return pca_images[0]

def batch_segmentation(model, x):
    _, _, x = model(x)  # Forward pass
    x = x.detach().cpu().numpy()  # Keep batch dimension
    x[x <= 0] = 0
    x[x > 0] = 1
    batch_size = x.shape[0]
    seg_masks = []
    for i in range(batch_size):
        feature_maps = np.transpose(x[i], (2, 0, 1))  # Convert to (C, H, W)
        seg = apply_pca_segmentation(feature_maps, n_components=1)
        thresh = threshold_otsu(seg)
        seg = seg > thresh
        seg_masks.append(seg)
    return np.array(seg_masks)

def compute_iou(predictions, targets):
    intersection = np.logical_and(predictions == 1, targets == 1).sum().item()
    union = np.logical_or(predictions == 1, targets == 1).sum().item()
    return intersection / union if union != 0 else float('nan')

def evaluate_model(model, dataloader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            predictions = batch_segmentation(model, images)
            ious.append(compute_iou(predictions, targets.detach().cpu().numpy()))
    return np.nanmean(ious)

def get_data_matek():
    data_path = "/lustre/groups/aih/michael.deutges/Datasets/Matek_Segmentation/image/"
    label_path = "/lustre/groups/aih/michael.deutges/Datasets/Matek_Segmentation/masks/"
    image_paths = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.jpg')]
    seg_paths = [os.path.join(label_path, file) for file in os.listdir(data_path) if file.endswith('.jpg')]
    class_labels = [os.path.basename(file).split('_')[0] for file in image_paths]
    unique_labels = sorted(set(class_labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    one_hot_labels = [np.eye(len(unique_labels))[label_to_index[label]] for label in class_labels]
    return image_paths, seg_paths, one_hot_labels, len(unique_labels)

def get_data_raabin():
    data_path = "/lustre/groups/aih/michael.deutges/Datasets/Raabin/images"
    label_path = "/lustre/groups/aih/michael.deutges/Datasets/Raabin/GT_masks"

    image_paths = []
    seg_paths = []
    class_labels = []
    
    # Get class names from folder structure
    class_folders = sorted(os.listdir(data_path))
    label_to_index = {cls: idx for idx, cls in enumerate(class_folders)}

    for cls in class_folders:
        cls_image_path = os.path.join(data_path, cls)
        cls_label_path = os.path.join(label_path, cls)

        if not os.path.isdir(cls_image_path):
            continue  # Skip non-folder files

        for file in os.listdir(cls_image_path):
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(cls_image_path, file))
                seg_paths.append(os.path.join(cls_label_path, file) if os.path.exists(os.path.join(cls_label_path, file)) else None)
                class_labels.append(cls)

    # Convert class labels to one-hot encoding
    unique_labels = sorted(label_to_index.keys())
    one_hot_labels = [np.eye(len(unique_labels))[label_to_index[label]] for label in class_labels]

    return image_paths, seg_paths, one_hot_labels, len(unique_labels)

def few_shot_sampling(X, y, z, samples_per_class):
    # Group data by class index
    class_dict = defaultdict(list)
    for img, seg, label in zip(X, y, z):
        class_idx = np.argmax(label)  # Get class index from one-hot encoding
        class_dict[class_idx].append((img, seg, label))  # Store image, mask, and one-hot label

    # Sample the required number of images per class
    few_shot_images, few_shot_masks, few_shot_labels = [], [], []

    for cls, samples in class_dict.items():
        np.random.shuffle(samples)  # Shuffle for randomness
        selected_samples = samples[:min(samples_per_class, len(samples))]  # Take only available samples

        for img, seg, label in selected_samples:
            few_shot_images.append(img)
            few_shot_masks.append(seg)
            few_shot_labels.append(label)

    return np.array(few_shot_images), np.array(few_shot_masks), np.array(few_shot_labels)

# CONFIGURATION OF EXPERIMENT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_paths, seg_paths, class_labels, out_dim = get_data_matek()
#image_paths, seg_paths, class_labels, out_dim = get_data_raabin()

X = np.asarray(image_paths)
y = np.asarray(seg_paths)
z = np.asarray(class_labels)

transform = transforms.Compose([
    transforms.RandomRotation([0, 360]),#transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
#mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
#mean=[0.696521,  0.5404207, 0.5858027], std=[0.1531189, 0.1879163, 0.08928457] Raabin
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
exp_name = "Matek_max_32_lr0001_focal"
channel_n = 32
EPOCHS = 128
weak=True
few_shot=False
samples_per_class = 10
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
output_path = "output3/"+exp_name
os.makedirs(output_path, exist_ok=True)

#TRAINING 5 FOLD
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    #if fold >=1:
    #    break
    print(f'Fold {fold + 1}/5')
    X_train, X_test = X[train_idx], X[test_idx] # images
    y_train, y_test = y[train_idx], y[test_idx] # segmentation masks
    z_train, z_test = z[train_idx], z[test_idx] # class labels

    # Few Shot
    if few_shot==True:
        X_train, y_train, z_train = few_shot_sampling(X_train, y_train, z_train, samples_per_class)

    if weak==False:
        train_dataset = Datasets.ContrastiveDataset(X_train, transform, resize=64)
        val_dataset = Datasets.ContrastiveDataset(X_test, test_transform, resize=64)
    else:
        train_dataset = Datasets.ClassDataset(X_train,z_train, transform, resize=64)
        val_dataset = Datasets.ClassDataset(X_test,z_test, test_transform, resize=64)
    test_dataset = Datasets.SegDataset(X_test, y_test, test_transform, resize=64)

    train_loader = data.DataLoader(train_dataset, batch_size=32)
    val_loader = data.DataLoader(val_dataset, batch_size=32)
    test_loader = data.DataLoader(test_dataset, batch_size=32)

    #model = Models.NCA(channel_n=channel_n, hidden_size=16, device=device, dropout=0, fire_rate=0.5, steps=32)
    model = Models.MaxNCA(channel_n=channel_n, hidden_size=32, device=device, fire_rate=0.5, steps=32, out_dim=out_dim)
    if few_shot==True:
        model.load_state_dict(torch.load("model_Soft-1x1conv-128a-001lr-64e-Focal.pth"))
        #model.load_state_dict(torch.load("output2/mean_32_unsupervised/model_fold1_mean_32_unsupervised.pth"))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9999)
    if weak == False:
        loss_fn = Losses.NTXentLoss()
    else:
        #loss_fn = torch.nn.CrossEntropyLoss() 
        loss_fn = Losses.FocalLoss()
    train_losses = []  # Store training loss per epoch
    val_losses = []
    iou_scores = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        if weak==False:
            for i, tdata in enumerate(train_loader):
                view1, view2 = tdata
                view1, view2 = view1.to(device), view2.to(device)
                
                optimizer.zero_grad()
                _,out1, _ = model(view1)
                _,out2, _ = model(view2)
                loss = loss_fn(out1.float(), out2.float())
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
        else:
            for i, tdata in enumerate(train_loader):
                inputs,labels = tdata
                inputs,labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                out, _, _ = model(inputs)
                loss = loss_fn(out.float(), labels.float())
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
        epoch_train_loss = running_loss / (i + 1)
        train_losses.append(epoch_train_loss)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            if weak == False:
                for j, vdata in enumerate(val_loader):
                    view1, view2 = vdata
                    view1, view2 = view1.to(device), view2.to(device)
                    _,out1, _ = model(view1)
                    _,out2, _ = model(view2)
                    vloss = loss_fn(out1.float(), out2.float())
                    val_loss += vloss.item()
            else:
                for j, vdata in enumerate(val_loader):
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    vout, _, _ = model(vinputs)
                    vloss = loss_fn(vout.float(), vlabels.float())
                    val_loss += vloss.item()

        epoch_val_loss = val_loss / (j + 1)
        val_losses.append(epoch_val_loss)

        print(f'Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
        
        avg_iou = evaluate_model(model, test_loader, device=device)
        iou_scores.append(avg_iou)
    
    model_save_path = os.path.join(output_path, f"model_fold{fold + 1}_{exp_name}.pth")
    torch.save(model.state_dict(), model_save_path)

    # Plot training & validation loss + IoU scores
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot Losses (Left Y-Axis)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, EPOCHS + 1), train_losses, marker='o', linestyle='-', color='red', label='Train Loss')
    ax1.plot(range(1, EPOCHS + 1), val_losses, marker='s', linestyle='--', color='darkred', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create second y-axis for IoU scores
    ax2 = ax1.twinx()
    ax2.set_ylabel('IoU Score', color='tab:blue')
    ax2.plot(range(1, EPOCHS + 1), iou_scores, marker='s', linestyle='--', color='darkblue', label='Val IoU')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Add legends
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    # Title and grid
    plt.title(f'Training & Validation Loss + IoU - Fold {fold + 1}')
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"loss_iou_plot_fold{fold + 1}_{exp_name}.png"))
    plt.show()
    

    avg_iou = evaluate_model(model, test_loader, device=device)
    results.append({"fold": fold + 1, "iou": avg_iou})
    print(f'Fold {fold + 1} - IoU: {avg_iou:.4f}')

results_save_path = os.path.join(output_path, f"results_{exp_name}.txt")
with open(results_save_path, "w") as f:
    for result in results:
        f.write(f"Fold {result['fold']} - IoU: {result['iou']:.4f}\n")










