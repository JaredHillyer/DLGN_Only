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

def get_data_MLL():
    image_dir="/lustre/groups/labs/marr/qscd01/workspace/raheleh.salehi/MRCNN-leukocyte/MRCNN-leukocyte/data/img_AML_MLL"
    mask_dir = "/lustre/groups/labs/marr/qscd01/workspace/raheleh.salehi/MRCNN-leukocyte/MRCNN-leukocyte/data/mask_AML_MLL"
    keyword="_Gal-"
    image_files = []
    mask_files = []

    # Iterate over images and filter by keyword
    for filename in os.listdir(image_dir):
        if keyword in filename:
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)  # Assuming masks have the same filename
            
            # Check if the corresponding mask exists
            if os.path.exists(mask_path):
                image_files.append(image_path)
                mask_files.append(mask_path)

    return image_files, mask_files


# CONFIGURATION
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
#image_paths, seg_paths, class_labels, out_dim = get_data_matek()
image_paths, seg_paths = get_data_MLL()
print(f"Found {len(image_paths)} images and {len(seg_paths)} masks.")

X = np.asarray(image_paths)
y = np.asarray(seg_paths)
#z = np.asarray(class_labels)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create dataset and dataloader
test_dataset = Datasets.SegDataset(X, y, test_transform, resize=64)
test_loader = data.DataLoader(test_dataset, batch_size=32)

for fold in range(5):
    # Load model
    model = Models.MeanNCA(channel_n=32, hidden_size=32, device=device, fire_rate=0.5, steps=32, out_dim=5)
    model.load_state_dict(torch.load("output3/Raabin_mean_32_lr0001/model_fold"+str(fold+1)+"_Raabin_mean_32_lr0001.pth"))
    #model.load_state_dict(torch.load("output2/Matek_mean_32_no-color/model_fold"+str(fold+1)+"_Matek_mean_32_no-color.pth"))
    
    model.to(device)

    # Evaluate model
    model.eval()
    avg_iou = evaluate_model(model, test_loader, device=device)
    print(f'Model IoU on full dataset: {avg_iou:.4f}')






























'''
#IMPORTS
import os
import torch
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import src.Datasets as Datasets
import src.Models as Models
import src.Losses as Losses
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#CONFIGURATION OF EXPERIMENT
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#DATALOADING
data_path="/lustre/groups/aih/michael.deutges/Datasets/Matek_Segmentation/image/"
label_path="/lustre/groups/aih/michael.deutges/Datasets/Matek_Segmentation/masks/"
image_paths = []
seg_paths = []

for file in os.listdir(data_path):
    if file.endswith('.jpg'):
        image_path = os.path.join(data_path, file)
        seg_path = os.path.join(label_path, file)
        image_paths.append(image_path)
        seg_paths.append(seg_path)  

X=np.asarray(image_paths)
y=np.asarray(seg_paths)
transform = transforms.Compose([
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

test_dataset = Datasets.SegDataset(X,y, transform , resize=64)
test_loader = data.DataLoader(test_dataset, batch_size=16)

# EXPERIMENT SETUP
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

channel_n=5
model = Models.NCA(
    channel_n=channel_n, 
    hidden_size=16, 
    device=device, 
    dropout=0, 
    fire_rate=0.5, 
    steps=16, 
)
model.load_state_dict(torch.load("/home/aih/michael.deutges/NCA_weak_segmentation/model_5c_16s_16h_mean_aug",map_location=torch.device('cpu'),weights_only=False))
model.eval()#,map_location=torch.device('cpu'))
model.to(device)
'''
'''
iterable = iter(test_loader)
dice_scores=[]
iou_scores=[]
j=-1
for i in enumerate(test_loader):
    j=j+1
    x,s=next(iterable)
    inputs=x.to(device)
    target=s.to(device)
    _,x=model(inputs)
    x=x.detach()#.cpu().numpy()
    #x=cv2.GaussianBlur(x, (3,3), 0)
    x[x<=0]=0
    x[x>0]=1
    input=x[:,:,:,6]
    


    n=30
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(int(0.5*cols * 2), int(0.5*rows * 2)))

    # Flatten axes array for easy iteration (if needed)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            if i ==0:
                ax.imshow(inputs[0].cpu())
                ax.axis("off") 
            else:
                ax.imshow(x[:,:,i+2])
                ax.axis("off") 
        else:
            ax.axis("off") 

    plt.tight_layout()
    plt.show()
    plt.savefig("test_plot.png")
    break
    



    input = torch.flatten(input)
    target = torch.flatten(target).to(device)
    intersection = (input * target).sum()
    dice = (2.*intersection)/(input.sum() + target.sum())
    iou = intersection/(input.sum() + target.sum()-intersection)
    #print(dice)
    print(iou)
    #dice_scores.append(dice.cpu())
    iou_scores.append(iou.cpu())

#dice_scores=np.asarray(dice_scores)
#print(dice_scores.mean())
iou_scores=np.asarray(iou_scores)
print("AVERAGE IOU:")
print(iou_scores.mean())
'''
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from scipy.stats import entropy
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy as entropy_filter
from skimage.morphology import disk
from skimage.filters.rank import entropy as entropy_filter
from skimage.morphology import disk

def apply_pca_segmentation_batch(feature_maps, n_components=3):
    N, C, H, W = feature_maps.shape  # Batch size, channels, height, width
    pca_images_batch = []
    pca_models = []
    
    for i in range(N):
        C, H, W = feature_maps[i].shape
        X = feature_maps[i].reshape(C, -1).T
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        pca_images = X_pca.T.reshape(n_components, H, W)
        
        pca_images_batch.append(pca_images)
        pca_models.append(pca)
    
    return np.array(pca_images_batch), pca_models

def cluster_feature_maps_batch(feature_maps, n_clusters=3):
    N, C, H, W = feature_maps.shape
    clustered_masks_batch = []
    
    for i in range(N):
        C, H, W = feature_maps[i].shape
        X = feature_maps[i].reshape(C, -1).T
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        clustered_mask = labels.reshape(H, W)
        clustered_masks_batch.append(clustered_mask)
    
    return np.array(clustered_masks_batch)

def compute_entropy(image, window_size=3):
    image = img_as_ubyte(image)
    return entropy_filter(image, disk(window_size))

def entropy_based_segmentation_batch(feature_maps, entropy_threshold=0.5):
    N, C, H, W = feature_maps.shape
    combined_maps_batch = []
    segmentation_masks_batch = []
    
    for i in range(N):
        entropy_values = np.array([np.mean(compute_entropy(fm)) for fm in feature_maps[i]])
        selected_maps = feature_maps[i][entropy_values < entropy_threshold * np.max(entropy_values)]
        
        if selected_maps.shape[0] == 0:
            print(f"No feature maps selected for sample {i}, try increasing the threshold.")
            combined_maps_batch.append(None)
            segmentation_masks_batch.append(None)
            continue
        
        combined_map = np.mean(selected_maps, axis=0)
        thresh = threshold_otsu(combined_map)
        segmentation_mask = combined_map > thresh
        
        combined_maps_batch.append(combined_map)
        segmentation_masks_batch.append(segmentation_mask)
    
    return np.array(combined_maps_batch), np.array(segmentation_masks_batch)


def compute_iou(predictions, targets):
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    
    :param predictions: Tensor of shape (N, H, W), predicted segmentation masks (0 for background, 1 for foreground).
    :param targets: Tensor of shape (N, H, W), ground truth segmentation masks.
    :return: Mean IoU.
    """
    intersection = torch.logical_and(predictions == 1, targets == 1).sum().item()
    union = torch.logical_or(predictions == 1, targets == 1).sum().item()
    
    return intersection / union if union != 0 else float('nan')

# Example usage
def evaluate_model(model, dataloader, device, threshold=0):
    """
    Evaluates the model on a given dataset and returns the average IoU.
    
    :param model: PyTorch segmentation model.
    :param dataloader: DataLoader for the test dataset.
    :param device: Torch device ("cuda" or "cpu").
    :param threshold: Threshold for converting probabilities/logits to binary mask.
    :return: Average IoU across the dataset.
    """
    model.eval()
    ious = []
    ious_inv = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            _,outputs = model(images)
            
            outputs=outputs[:,:,:,3]
            predictions = (outputs > threshold).int()  # Apply threshold to get binary mask
            predictions_inv = (outputs < threshold).int()
            # Example usage:
            #batch_feature_maps = predictions.permute(0,3,1,2).cpu()  # Example batch (N=8, C=16, H=128, W=128)
            #pca_results, _ = apply_pca_segmentation_batch(batch_feature_maps)
            #pca_results[pca_results<=0]=0
            #pca_results[pca_results>0]=1
            #clustered_masks = cluster_feature_maps_batch(batch_feature_maps)
            #combined_maps, segmentation_masks = entropy_based_segmentation_batch(batch_feature_maps)
            #predictions=torch.tensor(pca_results[:,0,:,:])
            batch_iou = compute_iou(predictions.cpu(), targets.cpu())
            ious.append(batch_iou)
            batch_iou_inv = compute_iou(predictions_inv.cpu(), targets.cpu())
            ious_inv.append(batch_iou_inv)
    
    return np.nanmean(ious),np.nanmean(ious_inv)


# Example call
avg_iou, avg_iou_inv = evaluate_model(model, test_loader, device="cuda")
print(avg_iou)
print(avg_iou_inv)
'''