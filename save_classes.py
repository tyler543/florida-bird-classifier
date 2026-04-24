# save_classes.py
# run once on windows pc to save class names to classes.txt

from torchvision import datasets
from config import DATA_DIR

dataset = datasets.ImageFolder(DATA_DIR)

with open('classes.txt', 'w') as f:
    for cls in dataset.classes:
        f.write(cls + '\n')

print(f"Saved {len(dataset.classes)} classes to classes.txt")
print(dataset.classes)