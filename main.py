import torch
from torchvision import transforms
from PIL import Image
import birder
from torch import nn
from torchvision import datasets
import time
import numpy as np
import cv2 as cv
from picamera2 import Picamera2
from config import *
from birdlib.ebird import get_species_info
from birdlib.supabase import insert_sighting
from gpiozero import Button
# Config 
model_name = MODEL_NAME
model_path = MODEL_PATH 
data_dir = DATA_DIR  # same folder used in training
device = DEVICE
# Button setup
button = Button(BUTTON_GPIO, pull_up=True)  # pull-up resistor

# variables
inference_hz = INFERENCE_HZ # inference per second
inference_interval = 1.0 / inference_hz
last_inference_time = 0.0
last_result = None
frames_probabilities = []
frame_avg_size = FRAME_AVERAGE_SIZE

# Load base model 
net, model_info = birder.load_pretrained_model(
    model_name,
    inference=False   # training-style model
)

# Get classes 
with open('classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
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

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
    )
)
picam2.start()
collecting = False
while True:
    frame = picam2.capture_array()
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

    button_held = button.is_pressed 

    if button_held and not collecting:
        # button just pressed, start collecting
        collecting = True
        frames_probabilities = []
        print("Button held — collecting frames...")

    elif button_held and collecting:
        # still holding, run inference on this frame
        now = time.perf_counter()
        if now - last_inference_time >= inference_interval:
            last_inference_time = now

            start_time = time.perf_counter()
            img = Image.fromarray(frame[:,:,::-1]).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = net(x)
                probs = torch.softmax(logits, dim=1)[0]

            inference_time = time.perf_counter() - start_time

            top_probs, top_indices = torch.topk(probs, TOP_N)
            top_probs_renorm = top_probs / top_probs.sum()
            frames_probabilities.append(top_probs_renorm)

            print(f"Frame {len(frames_probabilities)}/{FRAME_AVERAGE_SIZE} captured")

            if len(frames_probabilities) >= FRAME_AVERAGE_SIZE:
                avg_probs = torch.stack(frames_probabilities).mean(dim=0)
                predicted_species = classes[top_indices[avg_probs.argmax()]]
                confidence = avg_probs.max().item()

                print("\n--- Inference Result ---")
                print(f"Top {TOP_N} predictions:")
                for i, p in zip(top_indices, avg_probs):
                    print(f"{classes[i]}: {p.item():.4f}")
                print(f"Predicted class: {predicted_species} with confidence {confidence:.4f}")

                top_5 = {classes[i]: p.item() for i, p in zip(top_indices, avg_probs)}

                if predicted_species != "unknown":
                    ebird_info = get_species_info(predicted_species)
                    insert_sighting(
                        predicted_species=predicted_species,
                        confidence=confidence,
                        top_5=top_5,
                        ebird_info=ebird_info
                    )
                else:
                    print("Unknown species — skipping eBird and Supabase")

                # wait for button release before allowing next prediction
                collecting = False
                frames_probabilities = []

    elif not button_held and collecting:
        # let go early
        if len(frames_probabilities) < FRAME_AVERAGE_SIZE:
            print(f"Button released too early — hold for the full 3 seconds ({FRAME_AVERAGE_SIZE} frames needed, got {len(frames_probabilities)})")
        collecting = False
        frames_probabilities = []

# cleanup
picam2.stop()
cv.destroyAllWindows()