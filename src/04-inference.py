"""
04-inference.py

Legal Text Decoder – inference (basic)

- best_model.pt + vocab.txt betöltése
- checkpoint config alapján felépíti a megfelelő modellt (mlp/cnn/bilstm_attn)
- új, "unseen" szövegre is tud predikciót adni

Futtatás példák:
1) Egy szöveg:
   python src/04-inference.py --text "A szolgáltatás használatához internetkapcsolat szükséges."

2) CSV fájl sok szöveggel:
   python src/04-inference.py --input_csv /app/data/my_texts.csv --text_col text --n 20

   (Ha van label oszlopod is, add meg:)
   python src/04-inference.py --input_csv my.csv --text_col text --label_col label --n 50

3) Default (ha nincs input): test.csv-ből vesz N sort:
   python src/04-inference.py --n 10

Output:
- logol N példát
- ment CSV-t: processed_dir/inference_predictions.csv
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch

from utils import (
    setup_logger,
    load_config,
    load_vocab,
    encode_text,
    TinyLegalTextModel,
    TinyLegalTextCNN,
    TinyLegalTextBiLSTMAttn,
)

logger = setup_logger(__name__)


def build_model_from_checkpoint(ckpt_cfg: dict, vocab_size: int, pad_idx: int):
    model_type = str(ckpt_cfg.get("model_type", "mlp")).lower()

    embed_dim = int(ckpt_cfg.get("embed_dim", 64))
    dropout = float(ckpt_cfg.get("dropout", 0.2))
    num_classes = int(ckpt_cfg.get("num_classes", 5))

    # MLP
    hidden_dim = int(ckpt_cfg.get("hidden_dim", 32))

    # CNN
    cnn_num_filters = int(ckpt_cfg.get("cnn_num_filters", 64))
    cnn_kernel_sizes = ckpt_cfg.get("cnn_kernel_sizes", [3, 4, 5])
    cnn_kernel_sizes = tuple(int(k) for k in cnn_kernel_sizes)

    # BiLSTM+Attn
    lstm_hidden_dim = int(ckpt_cfg.get("lstm_hidden_dim", 64))
    lstm_num_layers = int(ckpt_cfg.get("lstm_num_layers", 1))
    bidirectional = bool(ckpt_cfg.get("bidirectional", True))

    if model_type == "cnn":
        return TinyLegalTextCNN(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_filters=cnn_num_filters,
            kernel_sizes=cnn_kernel_sizes,
            num_classes=num_classes,
            pad_idx=pad_idx,
            dropout=dropout,
        )

    if model_type == "bilstm_attn":
        return TinyLegalTextBiLSTMAttn(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            num_classes=num_classes,
            pad_idx=pad_idx,
            dropout=dropout,
            bidirectional=bidirectional,
            num_layers=lstm_num_layers,
        )

    return TinyLegalTextModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        pad_idx=pad_idx,
        dropout=dropout,
    )


def predict_single(model, device, text: str, word2idx: dict, max_len: int):
    ids = encode_text(text, word2idx, max_len)
    x = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
    return pred, conf, probs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None, help="Egy darab input szöveg inference-hez.")
    p.add_argument("--input_csv", type=str, default=None, help="CSV fájl sok szöveggel.")
    p.add_argument("--text_col", type=str, default="text", help="CSV-ben a szöveg oszlop neve.")
    p.add_argument("--label_col", type=str, default=None, help="(opcionális) CSV-ben a label oszlop neve.")
    p.add_argument("--n", type=int, default=10, help="Hány példát logoljon / dolgozzon fel.")
    return p.parse_args()


def main():
    logger.info("==== Legal Text Decoder - INFERENCE START ====")

    args = parse_args()
    cfg = load_config()

    processed_dir = Path(cfg.PROCESSED_DATA_DIR)
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"

    default_test_path = processed_dir / "test.csv"
    vocab_path = processed_dir / "vocab.txt"
    model_path = models_dir / "best_model.pt"

    # --- CONFIGURATION (tömör) ---
    logger.info("CONFIGURATION")
    logger.info(f"  PROCESSED_DATA_DIR={processed_dir}")
    logger.info(f"  MODELS_DIR={models_dir}")
    logger.info(f"  input_text={'YES' if args.text else 'NO'} | input_csv={args.input_csv}")
    logger.info(f"  n={args.n}")

    # --- Checks ---
    if not vocab_path.exists():
        logger.error(f"Missing vocab: {vocab_path}")
        return
    if not model_path.exists():
        logger.error(f"Missing model: {model_path}")
        return

    # --- Load vocab ---
    word2idx = load_vocab(vocab_path)
    vocab_size = len(word2idx)
    pad_idx = int(word2idx.get("<pad>", 0))

    # --- Device ---
    device = torch.device( "cpu")

    # --- Load checkpoint ---
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        ckpt_cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    else:
        state = ckpt
        ckpt_cfg = {}

    model_type = ckpt_cfg.get("model_type", getattr(cfg, "MODEL_TYPE", "mlp"))
    max_len = int(ckpt_cfg.get("max_len", getattr(cfg, "MAX_LEN", 128)))
    num_classes = int(ckpt_cfg.get("num_classes", getattr(cfg, "NUM_CLASSES", 5)))

    logger.info("CHECKPOINT INFO")
    logger.info(f"  model_path={model_path}")
    logger.info(f"  model_type={model_type} | max_len={max_len} | num_classes={num_classes}")
    logger.info(f"  vocab_size={vocab_size} | pad_idx={pad_idx}")
    logger.info(f"  device={device}")

    # --- Build model ---
    model = build_model_from_checkpoint(ckpt_cfg, vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    model.load_state_dict(state)
    model.eval()

    # --- Input resolution ---
    rows = []

    if args.text is not None:
        rows = [{"text": args.text}]
        source_name = "CLI_TEXT"

    elif args.input_csv is not None:
        in_path = Path(args.input_csv)
        if not in_path.exists():
            logger.error(f"Missing input_csv: {in_path}")
            return
        df_in = pd.read_csv(in_path)
        if args.text_col not in df_in.columns:
            logger.error(f"text_col '{args.text_col}' not found in CSV columns: {list(df_in.columns)}")
            return

        df_in = df_in.head(max(1, args.n)).copy()
        source_name = str(in_path)

        for _, r in df_in.iterrows():
            item = {"text": str(r[args.text_col])}
            if args.label_col and args.label_col in df_in.columns:
                item["label"] = int(r[args.label_col])
            rows.append(item)

    else:
        # fallback: processed test.csv
        if not default_test_path.exists():
            logger.error(f"No input provided and default test.csv missing: {default_test_path}")
            return
        df = pd.read_csv(default_test_path).head(max(1, args.n))
        source_name = str(default_test_path)

        for _, r in df.iterrows():
            rows.append({"text": str(r["text"]), "label": int(r["label"]) if "label" in df.columns else None})

    logger.info("INFERENCE")
    logger.info(f"  source={source_name} | examples={len(rows)}")

    # --- Run inference ---
    out_records = []
    for i, item in enumerate(rows):
        text = str(item.get("text", ""))
        true_label = item.get("label", None)

        pred_label, conf, probs = predict_single(model, device, text, word2idx, max_len=max_len)
        pred_rating = pred_label + 1
        true_rating = (int(true_label) + 1) if true_label is not None else None

        snippet = (text[:200] + "...") if len(text) > 200 else text

        # log only compact example lines
        if true_label is None:
            logger.info(f"  [{i}] pred={pred_label} (rating={pred_rating}) | conf={conf:.3f} | text='{snippet}'")
        else:
            logger.info(
                f"  [{i}] true={true_label} (rating={true_rating}) | "
                f"pred={pred_label} (rating={pred_rating}) | conf={conf:.3f} | text='{snippet}'"
            )

        rec = {
            "text": text,
            "pred_label": pred_label,
            "pred_rating": pred_rating,
            "confidence": conf,
        }
        if true_label is not None:
            rec["true_label"] = int(true_label)
            rec["true_rating"] = true_rating

        # (opcionális) probs oszlopok
        for c in range(num_classes):
            rec[f"prob_{c}"] = float(probs[c])

        out_records.append(rec)

    # --- Save output CSV ---
    out_path = processed_dir / "inference_predictions.csv"
    pd.DataFrame(out_records).to_csv(out_path, index=False)
    logger.info("OUTPUT")
    logger.info(f"  saved_predictions_csv={out_path}")

    logger.info("==== Legal Text Decoder - INFERENCE DONE ====")


if __name__ == "__main__":
    main()
