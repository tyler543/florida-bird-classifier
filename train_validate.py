import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import birder
from config import *
from plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_results_table,
    plot_sample_predictions,
)

# pull settings from config
data_dir   = DATA_DIR
batch_size = BATCH_SIZE
epochs     = EPOCHS
lr         = LR
seed       = SEED
device     = DEVICE
save_path  = MODEL_PATH

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs("models", exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

torch.manual_seed(seed)

# get input size and normalization stats from the pretrained model
net, model_info = birder.load_pretrained_model(MODEL_NAME, inference=False)
size         = birder.get_size_from_signature(model_info.signature)
rgb_stats    = model_info.rgb_stats
degrade_size = DEGRADE_SIZE

# train: downsample then upscale to simulate low-res input, plus basic augmentation
train_transform = transforms.Compose([
    transforms.Resize(degrade_size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=rgb_stats["mean"], std=rgb_stats["std"])
])

# val: clean resize/crop only, no augmentation
val_transform = transforms.Compose([
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=rgb_stats["mean"], std=rgb_stats["std"])
])

# no normalization — only used for displaying images in the predictions plot
display_transform = transforms.Compose([
    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(size),
])

# split dataset 80/20 using a fixed seed so the split is reproducible
train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_transform)
val_dataset_full   = datasets.ImageFolder(root=data_dir, transform=val_transform)

total_images = len(train_dataset_full)
train_size   = int(0.8 * total_images)
val_size     = total_images - train_size

generator = torch.Generator().manual_seed(seed)
perm = torch.randperm(total_images, generator=generator).tolist()
train_indices = perm[:train_size]
val_indices   = perm[train_size:]

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset   = Subset(val_dataset_full,   val_indices)

print(f"\nTotal images:      {total_images}")
print(f"Training images:   {train_size}")
print(f"Validation images: {val_size}")
print(f"Train / Val split: 80% / 20% (seed={seed})\n")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=True)

# swap the final layer to match our number of bird species
num_classes = len(train_dataset_full.classes)
class_names = train_dataset_full.classes
print("Classes:", class_names)

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)


def run_val_pass(model, loader, n_batches=None):
    # runs inference on the val set and returns loss, accuracy, and raw predictions
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(loader):
            if n_batches is not None and i >= n_batches:
                break
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = running_loss / total if total > 0 else float("nan")
    accuracy = correct / total * 100 if total > 0 else 0.0
    return avg_loss, accuracy, all_preds, all_labels


def measure_inference_time(model, loader, n_samples=100):
    # times n_samples individual forward passes and returns mean and std in ms
    model.eval()
    times = []
    count = 0
    with torch.no_grad():
        for imgs, _ in loader:
            for i in range(imgs.size(0)):
                img = imgs[i:i+1].to(device)
                t0 = time.perf_counter()
                model(img)
                times.append((time.perf_counter() - t0) * 1000)
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
    return float(np.mean(times)), float(np.std(times))


# baseline accuracy before any training starts
print("Computing pretrained-baseline accuracy on val subset...")
_, baseline_acc, _, _ = run_val_pass(net, val_loader, n_batches=4)
print(f"Pretrained baseline Top-1 (val subset): {baseline_acc:.2f}%\n")

# training loop — validate after every epoch so we can plot the curves later
history = {"train_loss": [], "val_loss": [], "val_acc": []}

start_time = time.perf_counter()

for epoch in range(epochs):
    net.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    train_loss = running_loss / train_size
    history["train_loss"].append(train_loss)

    val_loss, val_acc, _, _ = run_val_pass(net, val_loader)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1:3d}/{epochs}  "
          f"train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  "
          f"val_acc={val_acc:.2f}%")

total_time = time.perf_counter() - start_time

torch.save(net.state_dict(), save_path)
print(f"\nModel saved to: {save_path}")
print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
print(f"Total training time: {total_time:.2f} s")

# collect predictions across the full val set for the confusion matrix
print("\nRunning full validation pass for confusion matrix...")
_, final_acc, all_preds, all_labels = run_val_pass(net, val_loader)


print("Measuring inference time...")
infer_mean_ms, infer_std_ms = measure_inference_time(net, val_loader)
size_mb = os.path.getsize(save_path) / (1024 ** 2)

print(f"\nGenerating plots to {PLOTS_DIR}/")
plot_training_curves(history, baseline_acc, epochs, PLOTS_DIR)
plot_confusion_matrix(all_labels, all_preds, class_names, PLOTS_DIR)
plot_results_table(baseline_acc, final_acc, size_mb, infer_mean_ms, infer_std_ms, PLOTS_DIR)
plot_sample_predictions(net, val_indices, data_dir,
                        display_transform, val_transform,
                        class_names, device, seed, PLOTS_DIR)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Pretrained baseline accuracy : {baseline_acc:.2f}%")
print(f"  Fine-tuned final accuracy    : {final_acc:.2f}%")
print(f"  Accuracy gain                : +{final_acc - baseline_acc:.2f}%")
print(f"  Model size                   : {size_mb:.1f} MB")
print(f"  Inference time               : {infer_mean_ms:.1f} +/- {infer_std_ms:.1f} ms")
print(f"  Total training time          : {total_time:.1f} s")
