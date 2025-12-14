"""
02-training.py

Legal Text Decoder - training

Feladatok:
- train/val CSV beolvasása
- baseline (szabály alapú) értékelés val-on
- vocab építés + mentés
- Deep modell tanítása (MODEL_TYPE: mlp/cnn/bilstm_attn)
- epochonként train/val loss+acc log
- early stopping (PATIENCE)
- legjobb val_loss checkpoint mentése: models/best_model.pt

LOGGING (grading-ready):
- CONFIGURATION
- DATA LOADING
- MODEL ARCHITECTURE + param counts
- TRAINING PROGRESS (epoch sorok)
- VALIDATION (epoch sorok része)
- CHECKPOINT mentés
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import (
    setup_logger,
    load_config,
    build_vocab,
    save_vocab,
    encode_text,
    TinyLegalTextModel,
    TinyLegalTextCNN,
    TinyLegalTextBiLSTMAttn,
)

logger = setup_logger(__name__)


# ---------------------------
# Baseline (rule-based)
# ---------------------------
NEGATIVE_KWS = [
    "köteles", "tilt", "megszünteti", "felelősség", "megszűnik",
    "nem vállal", "hibáért", "szankció", "megtagad", "korlátozza",
]
POSITIVE_KWS = [
    "lehetőség", "igénybe veheti", "választhat", "megújítható",
    "használható", "engedélyezett", "kedvezmény",
]


def avg_word_length(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def count_keywords(text: str, kws) -> int:
    t = text.lower()
    return sum(1 for kw in kws if kw in t)


def rule_based_predict(text: str) -> int:
    num_words = len(text.split())
    awl = avg_word_length(text)
    neg = count_keywords(text, NEGATIVE_KWS)
    pos = count_keywords(text, POSITIVE_KWS)

    score = 0.0
    if num_words < 25:
        score += 1.0
    elif num_words > 60:
        score -= 1.0

    if awl > 9:
        score -= 0.7
    elif awl < 6:
        score += 0.4

    score += pos * 0.5
    score -= neg * 0.8

    if score <= -1.2:
        return 0
    elif score <= -0.4:
        return 1
    elif score <= 0.4:
        return 2
    elif score <= 1.2:
        return 3
    else:
        return 4


# ---------------------------
# Dataset
# ---------------------------
class LegalTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, word2idx: dict, max_len: int):
        self.df = df.reset_index(drop=True)
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = int(row["label"])
        ids = encode_text(text, self.word2idx, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ---------------------------
# Helpers
# ---------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_from_cfg(cfg, vocab_size: int, pad_idx: int):
    model_type = getattr(cfg, "MODEL_TYPE", "mlp").lower()

    if model_type == "cnn":
        return TinyLegalTextCNN(
            vocab_size=vocab_size,
            embed_dim=cfg.EMBED_DIM,
            num_filters=cfg.CNN_NUM_FILTERS,
            kernel_sizes=tuple(cfg.CNN_KERNEL_SIZES),
            num_classes=cfg.NUM_CLASSES,
            pad_idx=pad_idx,
            dropout=getattr(cfg, "DROPOUT", 0.2),
        )
    elif model_type == "bilstm_attn":
        return TinyLegalTextBiLSTMAttn(
            vocab_size=vocab_size,
            embed_dim=cfg.EMBED_DIM,
            lstm_hidden_dim=cfg.LSTM_HIDDEN_DIM,
            num_classes=cfg.NUM_CLASSES,
            pad_idx=pad_idx,
            dropout=getattr(cfg, "DROPOUT", 0.3),
            bidirectional=getattr(cfg, "BIDIRECTIONAL", True),
            num_layers=getattr(cfg, "LSTM_NUM_LAYERS", 1),
        )
    else:
        return TinyLegalTextModel(
            vocab_size=vocab_size,
            embed_dim=cfg.EMBED_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_classes=cfg.NUM_CLASSES,
            pad_idx=pad_idx,
            dropout=getattr(cfg, "DROPOUT", 0.2),
        )


def param_counts(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable, trainable + non_trainable


def compute_sqrt_class_weights(labels, num_classes: int, device):
    counts = Counter(labels)
    total = sum(counts.values())
    raw = torch.tensor([total / max(1, counts.get(i, 0)) for i in range(num_classes)], dtype=torch.float32)
    weights = torch.sqrt(raw)          # smoothing
    weights = weights / weights.mean() # normalize
    return counts, weights.to(device)


# ---------------------------
# Train
# ---------------------------
def train():
    logger.info("==== Legal Text Decoder - TRAINING START ====")

    cfg = load_config()

    # --- CONFIGURATION (grading-ready) ---
    logger.info("CONFIGURATION")
    cfg_kv = {
        "NUM_EPOCHS": cfg.NUM_EPOCHS,
        "BATCH_SIZE": cfg.BATCH_SIZE,
        "LEARNING_RATE": cfg.LEARNING_RATE,
        "WEIGHT_DECAY": getattr(cfg, "WEIGHT_DECAY", 0.0),
        "GRAD_CLIP_NORM": getattr(cfg, "GRAD_CLIP_NORM", 1.0),
        "PATIENCE": getattr(cfg, "PATIENCE", 5),
        "SEED": getattr(cfg, "SEED", 42),
        "MAX_LEN": cfg.MAX_LEN,
        "MAX_VOCAB_SIZE": getattr(cfg, "MAX_VOCAB_SIZE", 20000),
        "MIN_FREQ": getattr(cfg, "MIN_FREQ", 1),
        "MODEL_TYPE": getattr(cfg, "MODEL_TYPE", "mlp"),
        "NUM_CLASSES": cfg.NUM_CLASSES,
        "EMBED_DIM": getattr(cfg, "EMBED_DIM", None),
        "HIDDEN_DIM": getattr(cfg, "HIDDEN_DIM", None),
        "CNN_NUM_FILTERS": getattr(cfg, "CNN_NUM_FILTERS", None),
        "CNN_KERNEL_SIZES": getattr(cfg, "CNN_KERNEL_SIZES", None),
        "LSTM_HIDDEN_DIM": getattr(cfg, "LSTM_HIDDEN_DIM", None),
        "LSTM_NUM_LAYERS": getattr(cfg, "LSTM_NUM_LAYERS", None),
        "BIDIRECTIONAL": getattr(cfg, "BIDIRECTIONAL", None),
        "DROPOUT": getattr(cfg, "DROPOUT", None),
        "USE_CLASS_WEIGHTS": getattr(cfg, "USE_CLASS_WEIGHTS", False),
        "SAVE_BASELINE_CSV": getattr(cfg, "SAVE_BASELINE_CSV", True),
    }
    for k in sorted(cfg_kv.keys()):
        logger.info(f"  {k}={cfg_kv[k]}")

    seed = getattr(cfg, "SEED", 42)
    set_seed(seed)

    processed_dir = Path(cfg.PROCESSED_DATA_DIR)
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    if not train_path.exists() or not val_path.exists():
        logger.error("DATA LOADING: train/val CSV missing. Run 01-data-preprocessing.py first.")
        return

    # --- DATA LOADING ---
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    logger.info("DATA LOADING")
    logger.info(f"  train_csv={train_path} | n={len(df_train)}")
    logger.info(f"  val_csv={val_path}     | n={len(df_val)}")
    logger.info(f"  train_label_dist={Counter(df_train['label'].tolist())}")
    logger.info(f"  val_label_dist={Counter(df_val['label'].tolist())}")

    # --- BASELINE ---
    baseline_preds = [rule_based_predict(t) for t in df_val["text"].tolist()]
    baseline_acc = (df_val["label"].values == np.array(baseline_preds)).mean()
    logger.info("BASELINE (rule-based)")
    logger.info(f"  val_accuracy={baseline_acc:.4f}")

    if getattr(cfg, "SAVE_BASELINE_CSV", True):
        baseline_out = processed_dir / "val_with_baseline.csv"
        df_val_out = df_val.copy()
        df_val_out["baseline_pred"] = baseline_preds
        df_val_out.to_csv(baseline_out, index=False)
        logger.info(f"  saved={baseline_out}")

    # --- VOCAB ---
    word2idx = build_vocab(
        df_train["text"].tolist(),
        max_vocab_size=getattr(cfg, "MAX_VOCAB_SIZE", 20000),
        min_freq=getattr(cfg, "MIN_FREQ", 1),
    )
    vocab_path = processed_dir / "vocab.txt"
    save_vocab(word2idx, vocab_path)
    pad_idx = word2idx["<pad>"]

    logger.info("VOCAB")
    logger.info(f"  vocab_size={len(word2idx)}")
    logger.info(f"  vocab_path={vocab_path}")
    logger.info(f"  pad_idx={pad_idx} | unk_idx={word2idx.get('<unk>', 1)}")

    # --- DataLoaders ---
    train_ds = LegalTextDataset(df_train, word2idx, cfg.MAX_LEN)
    val_ds = LegalTextDataset(df_val, word2idx, cfg.MAX_LEN)

    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---
    model = build_model_from_cfg(cfg, vocab_size=len(word2idx), pad_idx=pad_idx).to(device)
    trainable, non_trainable, total = param_counts(model)

    logger.info("MODEL ARCHITECTURE")
    logger.info(str(model))
    logger.info(f"  params_trainable={trainable} | params_non_trainable={non_trainable} | params_total={total}")
    logger.info(f"  device={device}")

    # --- Loss ---
    criterion = nn.CrossEntropyLoss()
    if getattr(cfg, "USE_CLASS_WEIGHTS", False):
        counts, weights = compute_sqrt_class_weights(df_train["label"].tolist(), cfg.NUM_CLASSES, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        logger.info("CLASS WEIGHTS")
        logger.info(f"  class_counts={counts}")
        logger.info(f"  class_weights={weights.detach().cpu().numpy().round(4).tolist()}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=getattr(cfg, "WEIGHT_DECAY", 0.0),
    )

    # --- Training loop ---
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_path = models_dir / "best_model.pt"

    patience = getattr(cfg, "PATIENCE", 5)
    pat_ctr = 0

    logger.info("TRAINING PROGRESS")
    epochs_ran = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        epochs_ran = epoch

        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_n = 0

        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(cfg, "GRAD_CLIP_NORM", 1.0))
            optimizer.step()

            bs = yb.size(0)
            train_loss_sum += loss.item() * bs
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_n += bs

        train_loss = train_loss_sum / max(1, train_n)
        train_acc = train_correct / max(1, train_n)

        # Val
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_n = 0

        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)

                bs = yb.size(0)
                val_loss_sum += loss.item() * bs
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_n += bs

        val_loss = val_loss_sum / max(1, val_n)
        val_acc = val_correct / max(1, val_n)

        # One line per epoch (grading-friendly)
        logger.info(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            pat_ctr = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_path": str(vocab_path),
                    "vocab_size": len(word2idx),
                    "config": {
                        "model_type": getattr(cfg, "MODEL_TYPE", "mlp"),
                        "max_len": cfg.MAX_LEN,
                        "num_classes": cfg.NUM_CLASSES,
                        "embed_dim": getattr(cfg, "EMBED_DIM", None),
                        "hidden_dim": getattr(cfg, "HIDDEN_DIM", None),
                        "cnn_num_filters": getattr(cfg, "CNN_NUM_FILTERS", None),
                        "cnn_kernel_sizes": getattr(cfg, "CNN_KERNEL_SIZES", None),
                        "lstm_hidden_dim": getattr(cfg, "LSTM_HIDDEN_DIM", None),
                        "lstm_num_layers": getattr(cfg, "LSTM_NUM_LAYERS", None),
                        "bidirectional": getattr(cfg, "BIDIRECTIONAL", None),
                        "dropout": getattr(cfg, "DROPOUT", None),
                        "pad_idx": pad_idx,
                    },
                },
                best_model_path,
            )
            logger.info(f"CHECKPOINT: saved best_model.pt (epoch={epoch}, val_loss={val_loss:.4f})")
        else:
            pat_ctr += 1
            if pat_ctr >= patience:
                logger.info(f"EARLY STOPPING: patience={patience} reached (best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})")
                break

    # --- Finish summary ---
    logger.info("TRAINING FINISHED")
    logger.info(f"  epochs_ran={epochs_ran}")
    logger.info(f"  best_epoch={best_epoch}")
    logger.info(f"  best_val_loss={best_val_loss:.4f}")
    logger.info(f"  best_model_path={best_model_path}")
    logger.info("==== Legal Text Decoder - TRAINING DONE ====")


if __name__ == "__main__":
    train()
