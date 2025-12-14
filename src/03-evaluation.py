"""
03-evaluation.py

Legal Text Decoder – értékelés a TEST seten.

- test.csv beolvasása
- vocab.txt betöltése
- best_model.pt betöltése (PyTorch 2.6: weights_only=False)
- checkpoint config alapján felépíti a megfelelő modellt (mlp/cnn)
- teszt loss + accuracy + classification report + confusion matrix log
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from utils import setup_logger, load_config, load_vocab, encode_text, TinyLegalTextModel, TinyLegalTextCNN, TinyLegalTextBiLSTMAttn

logger = setup_logger(__name__)


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


def build_model_from_checkpoint(ckpt_cfg: dict, vocab_size: int, pad_idx: int):
    model_type = str(ckpt_cfg.get("model_type", "mlp")).lower()
    embed_dim = int(ckpt_cfg.get("embed_dim", 64))
    dropout = float(ckpt_cfg.get("dropout", 0.2))
    num_classes = int(ckpt_cfg.get("num_classes", 5))
    lstm_hidden_dim = int(ckpt_cfg.get("lstm_hidden_dim", 64))
    num_layers = int(ckpt_cfg.get("num_layers", 1))
    bidirectional = bool(ckpt_cfg.get("bidirectional", True))
    hidden_dim = int(ckpt_cfg.get("hidden_dim", 32))

    if model_type == "cnn":
        num_filters = int(ckpt_cfg.get("cnn_num_filters", 64))
        kernel_sizes = ckpt_cfg.get("cnn_kernel_sizes", [3, 4, 5])
        kernel_sizes = tuple(int(k) for k in kernel_sizes)

        return TinyLegalTextCNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            num_classes=num_classes,
            pad_idx=pad_idx,
            dropout=dropout,
        )
    
        
    elif model_type == "bilstm_attn":
        return TinyLegalTextBiLSTMAttn(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            num_classes=num_classes,
            pad_idx=pad_idx,
            dropout=dropout,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )


    return TinyLegalTextModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pad_idx=pad_idx,
        dropout=dropout,
    )


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    n = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            bs = yb.size(0)
            total_loss += loss.item() * bs
            n += bs

            preds = logits.argmax(dim=1)
            y_true.append(yb.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return total_loss / max(1, n), float((y_true == y_pred).mean()), y_true, y_pred


def main():
    logger.info("==== Legal Text Decoder - EVALUATION (TEST) START ====")

    cfg = load_config()
    processed_dir = Path(cfg.PROCESSED_DATA_DIR)
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"

    test_path = processed_dir / "test.csv"
    vocab_path = processed_dir / "vocab.txt"
    best_model_path = models_dir / "best_model.pt"

    if not test_path.exists():
        logger.error(f"Hiányzik: {test_path}")
        return
    if not vocab_path.exists():
        logger.error(f"Hiányzik: {vocab_path}")
        return
    if not best_model_path.exists():
        logger.error(f"Hiányzik: {best_model_path}")
        return

    df_test = pd.read_csv(test_path)
    logger.info(f"Test samples: {len(df_test)}")
    logger.info(f"Test label eloszlás: {Counter(df_test['label'].tolist())}")

    word2idx = load_vocab(vocab_path)
    vocab_size = len(word2idx)
    pad_idx = word2idx.get("<pad>", 0)

    logger.info(f"Vocab méret: {vocab_size}")
    logger.info(f"pad_idx: {pad_idx}")

    device = torch.device("cpu")
    logger.info(f"Használt device: {device}")

    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    ckpt_cfg = {}
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config", None), dict):
        ckpt_cfg = ckpt["config"]

    # max_len a checkpointból (ha van)
    max_len = int(ckpt_cfg.get("max_len", getattr(cfg, "MAX_LEN", 128)))
    num_classes = int(ckpt_cfg.get("num_classes", getattr(cfg, "NUM_CLASSES", 5)))

    test_ds = LegalTextDataset(df_test, word2idx, max_len=max_len)
    test_dl = DataLoader(test_ds, batch_size=getattr(cfg, "BATCH_SIZE", 32), shuffle=False)

    model = build_model_from_checkpoint(ckpt_cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    logger.info("Best model weights betöltve.")

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_dl, device, criterion)
    logger.info(f"TEST loss: {test_loss:.4f} | TEST accuracy: {test_acc:.4f}")

    logger.info("Classification report (TEST):")
    rep = classification_report(
        y_true,
        y_pred,
        digits=4,
        target_names=[f"class_{i}" for i in range(num_classes)],
        zero_division=0,
    )
    for line in rep.splitlines():
        logger.info(line)

    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion matrix (rows=true, cols=pred):")
    logger.info("\n" + str(cm))

    logger.info("==== Legal Text Decoder - EVALUATION (TEST) DONE ====")


if __name__ == "__main__":
    main()
