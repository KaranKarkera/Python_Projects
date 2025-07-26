# %% [markdown]
# Binary Segmentation for Road Detection
# %pip install numpy torch torchvision accelerate torchinfo pillow pandas matplotlib seaborn 

# %%
import os
import pandas as pd
import numpy as np 
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch import Tensor
from accelerate import Accelerator
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
from IPython.display import clear_output
from tqdm import tqdm
from time import time

# %%
class CONFIG:
    USE_MIXED_PRECISION = "fp16"
    DOWNSCALE = None
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    EXTRA_LOSS_EPS = 1e-6
    SNS_STYLE = "darkgrid"
    BATCH_SIZE = 8
    SINGLE_NETWORK_TRAINING_EPOCHS = 15
    CE_VS_DICE_EVAL_EPOCHS = 15
    DELTA_BETA = 0.2

cfg = CONFIG()
accelerator = Accelerator(mixed_precision=cfg.USE_MIXED_PRECISION) if cfg.USE_MIXED_PRECISION else Accelerator()

# %%
datapath = "Path/to/your/dataset/cityscapes_data"
train_datapath = os.path.join(datapath, "train")
val_datapath = os.path.join(datapath, "val")
training_images_paths = [os.path.join(train_datapath, f) for f in os.listdir(train_datapath)]
validation_images_paths = [os.path.join(val_datapath, f) for f in os.listdir(val_datapath)]

print(f"Size of training: {len(training_images_paths)}")
print(f"Size of validation: {len(validation_images_paths)}")

global_step = 0

# %%
# Visualization of sample images (unchanged)
width = 4
height = 4
vis_batch_size = width * height
indexes = np.random.permutation(len(training_images_paths))[:vis_batch_size]

fig, axs = plt.subplots(height, width, sharex=True, sharey=True, figsize=(16, 8))
for i in range(vis_batch_size):
    img = torchvision.io.read_image(training_images_paths[indexes[i]])
    img = img.permute(1, 2, 0)
    y, x = i // width, i % width
    axs[y, x].imshow(img.numpy())
plt.tight_layout()

# %%
# Class definitions (unchanged)
idx_to_name = [ 'unlabeled','ego vehicle','rectification border', 'out of roi', 'static', 'dynamic','ground', 'road', 'sidewalk', 'parking', 'rail track', 'building', 'wall', 'fence','guard rail' , 'bridge','tunnel','pole', 'polegroup', 'traffic light', 'traffic sign' ,'vegetation', 'terrain', 'sky' ,'person', 'rider', 'car','truck' ,'bus', 'caravan','trailer', 'train' , 'motorcycle','bicycle','license plate']

idx_to_color = [[ 0,  0,  0], [ 0,  0,  0], [  0,  0,  0], [  0,  0,  0],[ 0,  0,  0],[111, 74,  0],[81,  0, 81] ,[128, 64,128],[244, 35,232],
                [250,170,160],[230,150,140],[70, 70, 70],[102,102,156],[190,153,153],[180,165,180],[150,100,100],[150,120, 90],[153,153,153],
                [153,153,153],[250,170, 30],[220,220,  0],[107,142, 35],[152,251,152],[ 70,130,180],[220, 20, 60],[255,  0,  0],[ 0,  0,142],
                [ 0,  0, 70],[ 0, 60,100],[ 0,  0, 90],[  0,  0,110],[ 0, 80,100],[  0,  0,230],[119, 11, 32],[  0,  0,142]]

ROAD_CLASS_IDX = 7  # Index of 'road' in class list

# %%
def preprocess_image(path: str, downscale_factor=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - input image (HxWx3)
    - binary mask (HxW): 1 for road, 0 otherwise
    """
    img = Image.open(path)
    width, height = img.size

    if downscale_factor:
        width, height = width // downscale_factor, height // downscale_factor
        img = img.resize((width, height))

    img = np.asarray(img)
    raw, mask = img[:, :width // 2, :], img[:, width // 2:, :]

    # Calculate distances to all class colors
    h, w, c = mask.shape
    distances = np.sum((mask.reshape(-1, c)[:, np.newaxis, :] - np.array(idx_to_color)) ** 2, axis=2)
    classes = np.argmin(distances, axis=1).reshape(h, w)

    # Create binary mask: road=1, not road=0
    binary_mask = (classes == ROAD_CLASS_IDX).astype(np.uint8)
    return raw, binary_mask

# %%
# Sample visualization
x, binary_mask = preprocess_image(training_images_paths[indexes[0]])
plt.subplot(1, 2, 1)
plt.imshow(x)
plt.title("Input Image")
plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title("Binary Road Mask")
plt.show()

# %%
import multiprocessing as mp

downscale_factor = cfg.DOWNSCALE

def process_image(path):
    X, Y = preprocess_image(path, downscale_factor=downscale_factor)
    X = torch.tensor(X / 255., dtype=torch.float32).permute(2, 0, 1)
    Y = torch.tensor(Y, dtype=torch.long)  # Binary mask
    return X, Y

def preprocess_with_multiprocessing(image_paths, desc="Processing"):
    results = []
    with mp.get_context("fork").Pool(processes=mp.cpu_count() - 1) as pool:
        for res in tqdm(pool.imap(process_image, image_paths), total=len(image_paths), desc=desc):
            results.append(res)
    return results

# Training images
train_results = preprocess_with_multiprocessing(training_images_paths, desc="Training Set")
X_train, Y_train = zip(*train_results)

# Validation images
val_results = preprocess_with_multiprocessing(validation_images_paths, desc="Validation Set")
X_val, Y_val = zip(*val_results)

# %%
class CityScapesDataset(Dataset):
    def __init__(self, X, Y, transform=None, target_transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

# %%
preprocess = transforms.Compose([
    transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
])

train_ds = CityScapesDataset(X_train, Y_train, transform=preprocess)
val_ds = CityScapesDataset(X_val, Y_val, transform=preprocess)

train_dataloader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=14)
val_dataloader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=14)

# %%
# Loss functions for binary segmentation
def dice_coeff(inp: Tensor, tgt: Tensor, eps=cfg.EXTRA_LOSS_EPS):
    # Flatten tensors
    inp = inp.view(-1)
    tgt = tgt.view(-1)
    
    intersection = (inp * tgt).sum()
    union = inp.sum() + tgt.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return dice

def dice_loss(input: Tensor, target: Tensor):
    return 1 - dice_coeff(torch.sigmoid(input), target)

def IoU_coeff(inp: Tensor, tgt: Tensor, eps=1e-6):
    inp = inp.view(-1)
    tgt = tgt.view(-1)
    
    intersection = (inp * tgt).sum()
    union = inp.sum() + tgt.sum() - intersection
    return (intersection + eps) / (union + eps)

def IoU_loss(inp: Tensor, tgt: Tensor):
    return 1 - IoU_coeff(torch.sigmoid(inp), tgt)

# %%
# Training function for binary segmentation
def train_model(model, device, train_dataloader, val_dataloader, epochs=10, lr=1e-4):
    global global_step
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy
    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)
    
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_dice, train_iou = 0.0, 0.0, 0.0
        num_samples = 0
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device).float()  # BCE requires float targets
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Calculate losses
                bce_loss = criterion(outputs.squeeze(1), masks)
                d_loss = dice_loss(outputs, masks)
                iou_loss_val = IoU_loss(outputs, masks)
                
                # Combined loss
                loss = bce_loss + d_loss
                
                accelerator.backward(loss)
                optimizer.step()
                
                # Update metrics
                bs = images.size(0)
                train_loss += loss.item() * bs
                train_dice += (1 - d_loss.item()) * bs
                train_iou += (1 - iou_loss_val.item()) * bs
                num_samples += bs
                
                pbar.set_postfix({
                    'loss': train_loss / num_samples,
                    'dice': train_dice / num_samples,
                    'iou': train_iou / num_samples
                })
                global_step += 1
        
        # Validation
        val_metrics = evaluate_model(model, val_dataloader, device)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss / num_samples,
            'train_dice': train_dice / num_samples,
            'train_iou': train_iou / num_samples,
            **val_metrics
        })
    
    return history

def evaluate_model(model, dataloader, device):
    model.eval()
    val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
    num_samples = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device).float()
            
            outputs = model(images)
            
            # Calculate losses
            bce_loss = criterion(outputs.squeeze(1), masks)
            d_loss = dice_loss(outputs, masks)
            iou_loss_val = IoU_loss(outputs, masks)
            loss = bce_loss + d_loss
            
            # Update metrics
            bs = images.size(0)
            val_loss += loss.item() * bs
            val_dice += (1 - d_loss.item()) * bs
            val_iou += (1 - iou_loss_val.item()) * bs
            num_samples += bs
    
    return {
        'val_loss': val_loss / num_samples,
        'val_dice': val_dice / num_samples,
        'val_iou': val_iou / num_samples
    }

# %%
def show_inference(batch, predictions):
    images, true_masks = batch
    batch_size = images.shape[0]
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    
    for i in range(batch_size):
        # Input image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = img * np.array(cfg.STD) + np.array(cfg.MEAN)  # Denormalize
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')
        
        # Ground truth
        gt_mask = true_masks[i].cpu().numpy()
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Prediction
        pred = torch.sigmoid(predictions[i])
        pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)  # Single output channel for binary segmentation

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits  # Output will be passed through sigmoid for probabilities
    
# %%
device = accelerator.device
model = UNet(n_channels=3).to(device)
summary(model, input_size=(1, 3, 256, 256))

# %%
# Training
history = train_model(
    model, 
    device, 
    train_dataloader, 
    val_dataloader,
    epochs=cfg.SINGLE_NETWORK_TRAINING_EPOCHS,
    lr=1e-4
)

# %%
# Save model
torch.save(model.state_dict(), "road_segmentation_unet.pth")

# %%
# Inference on validation set
model.eval()
batch = next(iter(val_dataloader))
with torch.no_grad():
    predictions = model(batch[0].to(device))
show_inference(batch, predictions)

# %%
# Inference on custom images
def run_inference_on_custom_images(model, image_paths, device="cuda"):
    model.eval()
    images_tensor = []
    original_images = []
    
    for path in image_paths:
        raw_img = Image.open(path).convert("RGB")
        original_img = np.array(raw_img)
        original_images.append(original_img)
        
        # Preprocess
        raw_img = raw_img.resize((256, 256))
        img_tensor = torch.tensor(np.array(raw_img) / 255., dtype=torch.float32).permute(2, 0, 1)
        img_tensor = transforms.Normalize(cfg.MEAN, cfg.STD)(img_tensor)
        images_tensor.append(img_tensor)
    
    images_tensor = torch.stack(images_tensor).to(device)
    
    with torch.no_grad():
        preds = model(images_tensor)
    
    # Plot results
    batch_size = len(image_paths)
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    
    for i in range(batch_size):
        # Original image
        axes[i, 0].imshow(original_images[i])
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')
        
        # Prediction
        pred = torch.sigmoid(preds[i])
        pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()
        axes[i, 1].imshow(pred_mask, cmap='gray')
        axes[i, 1].set_title("Road Prediction")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Test on sample images
custom_image_paths = [
    "/location/to/your/images.jpg,
    # Add more paths
]
run_inference_on_custom_images(model, custom_image_paths, device=device)
# %%
