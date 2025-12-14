from pathlib import Path


TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

MAX_LEN = 128
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 1

NUM_CLASSES = 5

NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
PATIENCE = 5
USE_CLASS_WEIGHTS = True
SEED = 42
WEIGHT_DECAY = 1e-4

EMBED_DIM = 64
HIDDEN_DIM = 32
DROPOUT = 0.4
GRAD_CLIP_NORM = 1.0

CNN_NUM_FILTERS = 64
CNN_KERNEL_SIZES = [3, 4, 5]

MODEL_TYPE = "bilstm_attn"

# LSTM
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 1
BIDIRECTIONAL = True


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR_SAVE = DATA_DIR / "raw" / "legal_text_decoder"
RAW_DATA_DIR = DATA_DIR / "raw" / "legal_text_decoder" / "legaltextdecoder"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "legal_text_decoder" 
MODEL_SAVE_PATH = PROJECT_ROOT / "model.pth"

# Data link
DATA_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1"


