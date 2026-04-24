# config.py
# Central configuration file — edit paths and settings here

import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.path.join(BASE_DIR, "florida_birds")
MODEL_PATH = os.path.join(BASE_DIR, "models", "florida_birds_v1_final.pth")
TEST_IMAGE = os.path.join(BASE_DIR, "test.jpg")

# Model
MODEL_NAME = "mobilenet_v4_s_il-common"

# Training
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
SEED = 42
DEGRADE_SIZE = 128

# Inference
INFERENCE_HZ = 1
TOP_N = 5

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"