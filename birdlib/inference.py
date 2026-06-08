import torch
import birder
from torch import nn
from PIL import Image
from torchvision import transforms

def load_classes(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]

def load_model(model_name, model_path, num_classes, device):
    net, model_info = birder.load_pretrained_model(
        model_name,
        inference=False
    )
    if hasattr(net, 'classifier'):
        if isinstance(net.classifier, nn.Sequential):
            in_features = net.classifier[-1].in_features
            net.classifier[-1] = nn.Linear(
                in_features,
                num_classes
            )
        else:
            in_features = net.classifier.in_features
            net.classifier = nn.Linear(
                in_features,
                num_classes
            )
    elif hasattr(net, 'fc'):
        in_features = net.fc.in_features
        net.fc = nn.Linear(
            in_features,
            num_classes
        )
    else:
        raise ValueError(
            "Model architecture not supported."
        )
    net.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )
    net.to(device)
    net.eval()
    return net, model_info

def build_transform(model_info):
    size = birder.get_size_from_signature(
        model_info.signature
    )
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
    return transform, size

def run_inference(frame, net, transform, device):
    img = Image.fromarray(
        frame[:, :, ::-1]
    ).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(x)
        probs = torch.softmax(
            logits,
            dim=1
        )[0]
    return probs

def average_frames(frames_probabilities):
    return torch.stack(
        frames_probabilities
    ).mean(dim=0)
    
def get_top_predictions(
    avg_probs,
    top_indices,
    classes,
    top_n
):
    predicted_species = classes[
        top_indices[avg_probs.argmax()]
    ]
    confidence = avg_probs.max().item()
    top_5 = {
        classes[i]: p.item()
        for i, p in zip(
            top_indices,
            avg_probs
        )
    }
    return predicted_species, confidence, top_5

def extract_topk_and_normalize(probs, top_n):
    top_probs, top_indices = torch.topk(probs,top_n)
    
    top_probs_renorm = top_probs / top_probs.sum()
    
    return top_probs_renorm, top_indices