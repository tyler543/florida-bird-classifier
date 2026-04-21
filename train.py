import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import birder
from birder.inference.classification import infer_image
import time
from config import *
# Config
data_dir = DATA_DIR
model_name = MODEL_NAME
batch_size = BATCH_SIZE
epochs = EPOCHS
lr = LR
device = DEVICE
save_path = MODEL_PATH

# check if save folder exists
os.makedirs("models", exist_ok=True)
start_time = time.perf_counter() # start time
# load pretrained model 
net, model_info = birder.load_pretrained_model(model_name, inference=False)

# get model input size & RGB stats
size = birder.get_size_from_signature(model_info.signature)
rgb_stats = model_info.rgb_stats

# Dataset
transform = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=rgb_stats["mean"], std=rgb_stats["std"])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# Replace final layer 
num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

if hasattr(net, 'classifier'):
    
    if isinstance(net.classifier, nn.Sequential):
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        # classifier is linear layer
        in_features = net.classifier.in_features
        net.classifier = nn.Linear(in_features, num_classes)

elif hasattr(net, 'fc'):
    # for models like ResNet
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    
else:
    raise ValueError("Model architecture not supported for replacing final layer.")

net = net.to(device)
# Loss & Optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

# training loop 
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

end_time = time.perf_counter() # end time
# save model 
torch.save(net.state_dict(), save_path)
print(f"Model saved as {save_path}")
print(f"Training time: {(end_time - start_time):.2f} seconds")