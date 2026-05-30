# config.py
# Central configuration file — edit paths and settings here

from dotenv import load_dotenv
import os
import torch

load_dotenv()

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

# Inference (current set up is 12 frames over 3 seconds)
INFERENCE_HZ = 4
TOP_N = 5
FRAME_AVERAGE_SIZE = 12 # amount of frames 
BUTTON_GPIO = 27 # BCM pin for the hold-to-infer button

# API Keys
EBIRD_API_KEY = os.getenv("EBIRD_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"