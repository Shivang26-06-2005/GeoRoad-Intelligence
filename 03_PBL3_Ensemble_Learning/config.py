import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "City_Scale Dataset", "data", "20cities")
SPLIT_PATH = os.path.join(PROJECT_ROOT, "City_Scale Dataset", "data", "data_split.json")

# Module paths
UNIT5_DIR = os.path.join(PROJECT_ROOT, "unit5_ensemble")
MODELS_DIR = os.path.join(UNIT5_DIR, "models")
OUTPUTS_DIR = os.path.join(UNIT5_DIR, "outputs")
PBL3_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pbl3_output")

# Experiment configuration
RANDOM_SEED = 42
SAMPLES_PER_REGION_TRAIN = 3000   # Balanced pixel samples per region
SAMPLES_PER_REGION_VAL = 1000
SAMPLES_PER_REGION_TEST = 1500

MAX_TRAIN_REGIONS = 20  # Representative diverse geographic training regions for rigorous SVM & ensemble training
TEST_REGION_ID = 9      # Canonical held-out test region for full 2D spatial visualization

# Feature definitions
FEATURE_NAMES = [
    "R", "G", "B",
    "HSV_H", "HSV_S", "HSV_V",
    "LAB_L", "LAB_A", "LAB_B",
    "Gray",
    "Sobel_X", "Sobel_Y", "Grad_Mag",
    "Laplacian",
    "Mean_5x5", "Mean_11x11",
    "Std_5x5", "Std_11x11"
]

