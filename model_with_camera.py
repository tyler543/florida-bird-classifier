import torch
from torchvision import transforms
from PIL import Image
import birder
from torch import nn
from torchvision import datasets
import time
import numpy as np
import cv2 as cv
from config import *
# Config 
model_name = MODEL_NAME
model_path = MODEL_PATH 
data_dir = DATA_DIR  # same folder used in training
device = DEVICE

# variables
inference_hz = INFERENCE_HZ # inference per second
inference_interval = 1.0 / inference_hz
last_inference_time = 0.0
last_result = None

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
    transforms.CenterCrop(size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=rgb_stats["mean"],
        std=rgb_stats["std"]
    )
])

# rename to your camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    
    now = time.perf_counter()
    if now - last_inference_time >= inference_interval:
        last_inference_time = now 
        
        start_time = time.perf_counter()
        
        img = Image.fromarray(frame[:,:,::-1]).convert("RGB")
        
        # img.save("debug_webcam_input.jpg")
        
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = net(x)
            probs = torch.softmax(logits, dim=1)[0]
        
        inference_time = time.perf_counter() - start_time
        
        last_result = probs.cpu()
        top_n = 5
        top_probs, top_indices = torch.topk(probs, top_n)
        top_probs_renorm = top_probs / top_probs.sum()
        
        print("Top predictions:")
        for i, p in zip(top_indices, top_probs_renorm):
            print(f"{classes[i]}: {p.item():.4f}")
            
        pred_idx = torch.argmax(probs).item()
        # best prediction
        print(f"\nPredicted class: {classes[pred_idx]}")
        print(f"Model inference time: {(inference_time):.2f} seconds") 
    # Preprocess frame
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()