import birder
from birder.inference.classification import infer_image

# Load a pretrained model for inference
# You can choose from the list of available models in birder.list_pretrained_models()
model_name = "mobilenet_v4_s_il-common"

(net, model_info) = birder.load_pretrained_model(model_name, inference=True)

# Get the image size the model was trained on
size = birder.get_size_from_signature(model_info.signature)

# Create an inference transform
transform = birder.classification_transform(size, model_info.rgb_stats)

image = "test.jpg"  # or a PIL image, must be loaded in RGB format
(out, _) = infer_image(net, image, transform)

# out is a NumPy array with shape of (1, 707), representing class probabilities.
# sort from best values to worst
probs = out[0]

# top N predictions
top_n = 5
top_indices = probs.argsort()[::-1][:top_n]

#map indices to class names
idx_to_class = {v: k for k, v in model_info.class_to_idx.items()}
top_classes = [idx_to_class[i] for i in top_indices]
top_probs = probs[top_indices]

top_probs = top_probs / top_probs.sum()

# predictions renormalized
for cls, p in zip(top_classes, top_probs):
    print(f"{cls}: {p:.4f}")