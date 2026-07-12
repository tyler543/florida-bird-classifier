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

# Inference
INFERENCE_HZ = 4
TOP_N = 5
FRAME_AVERAGE_SIZE = 12
BUTTON_GPIO = 23
INFERENCE_SKIP = 4    # run inference every N captured frames
CONF_THRESHOLD = 0.25 # minimum confidence to accept a live prediction

# API Keys
EBIRD_API_KEY = os.getenv("EBIRD_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TUNING_FILE = os.getenv("TUNING_FILE")

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = os.getenv("DEVICE_ID", "FEATHER-000")
HUD_SYNC_INTERVAL = 2.0  # seconds between Supabase HUD-settings polls