import torch
from torchvision import transforms
from PIL import Image
import birder
from torch import nn
from torchvision import datasets
import time
from config import *
# Config 
model_name = MODEL_NAME
model_path = MODEL_PATH
image_path = TEST_IMAGE
data_dir = DATA_DIR  # same folder used in training
device = DEVICE

# Load base model 
net, model_info = birder.load_pretrained_model(
    model_name,
    inference=False   # training-style model
)

# Get classes 
dataset = datasets.ImageFolder(root=data_dir)
classes = dataset.classes  # this will match the training folder order
num_classes = len(classes)

# recreate classifier head to match number of classes
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

# load trained weights
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()

# transforms (must match training) 
size = birder.get_size_from_signature(model_info.signature)
rgb_stats = model_info.rgb_stats

transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=rgb_stats["mean"],
        std=rgb_stats["std"]
    )
])

start_time = time.perf_counter()
#  Load & preprocess image 
img = Image.open(image_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
   
    logits = net(x)
    probs = torch.softmax(logits, dim=1)[0]
    
end_time = time.perf_counter()
# Print results
# Optionally, get top N predictions
top_n = 5
top_probs, top_indices = torch.topk(probs, top_n)

# Renormalize top probabilities
top_probs_renorm = top_probs / top_probs.sum()

print("Top predictions:")
for i, p in zip(top_indices, top_probs_renorm):
    print(f"{classes[i]}: {p.item():.4f}")

# best prediction
pred_idx = torch.argmax(probs).item()
print(f"\nPredicted class: {classes[pred_idx]}")
print(f"Model inference time: {(end_time - start_time):.2f} seconds")