import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import birder
import time
from config import *

# Config
data_dir = DATA_DIR
model_name = MODEL_NAME

# training config
batch_size = BATCH_SIZE
epochs = EPOCHS
lr = LR
seed = SEED 
device = DEVICE
save_path = MODEL_PATH

# Setup 
os.makedirs("models", exist_ok=True)
torch.manual_seed(seed)

start_time = time.perf_counter()

#  Load pretrained model
net, model_info = birder.load_pretrained_model(model_name, inference=False)

size = birder.get_size_from_signature(model_info.signature)
rgb_stats = model_info.rgb_stats

degrade_size = DEGRADE_SIZE

train_transform = transforms.Compose([
    transforms.Resize(degrade_size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=rgb_stats["mean"], std=rgb_stats["std"])
])

val_transform = transforms.Compose([
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=rgb_stats["mean"], std=rgb_stats["std"])
])

train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_transform)
val_dataset_full = datasets.ImageFolder(root=data_dir, transform=val_transform)

#  Dataset split (80 / 20)
total_images = len(train_dataset_full)
train_size = int(0.8 * total_images)
val_size = total_images - train_size

generator = torch.Generator().manual_seed(seed)
perm = torch.randperm(total_images, generator=generator).tolist()
train_dataset = Subset(train_dataset_full, perm[:train_size])
val_dataset = Subset(val_dataset_full, perm[train_size:])

print(f"\nTotal images: {total_images}")
print(f"Training images: {train_size}")
print(f"Validation images: {val_size}")
print(f"Train / Val split: 80% / 20% (seed={seed})\n")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)

#  Replace classifier 
num_classes = len(train_dataset_full.classes)
print("Classes:", train_dataset_full.classes)

if hasattr(net, "classifier"):
    if isinstance(net.classifier, nn.Sequential):
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = net.classifier.in_features
        net.classifier = nn.Linear(in_features, num_classes)
elif hasattr(net, "fc"):
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
else:
    raise RuntimeError("Unsupported model architecture")

net = net.to(device)

#  Loss & optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

#  Training loop 
for epoch in range(epochs):
    net.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / train_size
    print(f"Epoch {epoch+1}/{epochs} — Train Loss: {epoch_loss:.4f}")

#  Validation 
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = net(imgs)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_accuracy = correct / total * 100

#  Save & timing 
end_time = time.perf_counter()
torch.save(net.state_dict(), save_path)

print(f"\nModel saved to: {save_path}")
print(f"Validation accuracy: {val_accuracy:.2f}%")
print(f"Total training + validation time: {(end_time - start_time):.2f} seconds")