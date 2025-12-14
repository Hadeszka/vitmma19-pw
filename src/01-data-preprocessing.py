"""
01-data-preprocessing.py

Legal Text Decoder – Data Preprocessing

Feladatok:
- Nyers adatok letöltése egy URL-ről (ZIP) cfg.DATA_URL alapján.
- ZIP kicsomagolása a RAW_DATA_DIR_SAVE alá (ha kell).
- Label Studio JSON fájlok beolvasása rekurzívan (data.text + rating).
- (text, label) párok előállítása, label ∈ {0,1,2,3,4}.
- Split:
    * test adatok a "consensus" mappából jönnek,
    * consensus maradék mehet train/val-ba is,
    * BP17IB mappa teljes kihagyása (meta JSON), hibás tartalom miatt.
- Deduplikálás normalizált szöveg alapján.
- Mentés: train.csv, val.csv, test.csv a PROCESSED_DATA_DIR alá.
"""

from pathlib import Path
from collections import Counter
import io
import zipfile
import json

import requests
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import setup_logger, load_config

logger = setup_logger(__name__)


# -------------------------
# Helpers
# -------------------------
def normalize_text_basic(s: str) -> str:
    """Whitespace összehúzás + kisbetűsítés (case-insensitive dedup)."""
    if not isinstance(s, str):
        return ""
    s = " ".join(s.split())
    return s.lower()


def safe_json_load(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # fallback enc
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)


# -------------------------
# Download
# -------------------------
def download_raw_data_if_needed(cfg):
    """
    Ha a RAW_DATA_DIR (rekurzívan) nem tartalmaz JSON fájlokat,
    megpróbálja letölteni a datasetet a cfg.DATA_URL-ről ZIP formátumban,
    és kicsomagolja a RAW_DATA_DIR_SAVE alá.
    """
    raw_dir = Path(cfg.RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_dir_save = Path(getattr(cfg, "RAW_DATA_DIR_SAVE", cfg.RAW_DATA_DIR))
    raw_dir_save.mkdir(parents=True, exist_ok=True)

    existing_jsons = list(raw_dir.rglob("*.json"))
    if existing_jsons:
        logger.info(f"DATA: existing raw JSON files found: n={len(existing_jsons)} -> skipping download")
        return

    logger.info("DATA: no raw JSON found -> downloading dataset ZIP")

    resp = requests.get(cfg.DATA_URL, stream=True, timeout=120)
    resp.raise_for_status()
    content = resp.content

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(raw_dir_save)
    except zipfile.BadZipFile:
        out_path = raw_dir_save / "legal_text_decoder.json"
        out_path.write_bytes(content)


# -------------------------
# Label Studio reader
# -------------------------
def load_labelstudio_json(path: Path):
    """
    Label Studio JSON fájlból:
      - data.text
      - annotations[0].result[0].value.rating (1..5) vagy value.choices

    Visszatér: list[(text, label)] ahol label 0..4
    """
    data = safe_json_load(path)
    if not isinstance(data, list):
        return []

    samples = []
    for item in data:
        if not isinstance(item, dict):
            continue

        text = item.get("data", {}).get("text", "")
        if not text:
            continue

        annotations = item.get("annotations", [])
        if not annotations:
            continue

        result_list = annotations[0].get("result", [])
        if not result_list:
            continue

        value = result_list[0].get("value", {})
        rating = None

        if isinstance(value, dict) and "rating" in value:
            try:
                rating = int(value["rating"])
            except Exception:
                rating = None
        elif isinstance(value, dict) and "choices" in value and value["choices"]:
            try:
                rating = int(str(value["choices"][0]).split("-")[0])
            except Exception:
                rating = None

        if rating is None or rating < 1 or rating > 5:
            continue

        label = rating - 1
        samples.append((text, label))

    return samples


# -------------------------
# Main preprocessing
# -------------------------
def preprocess():
    logger.info("==== Legal Text Decoder - PREPROCESS START ====")

    cfg = load_config()

    # Config log (tömören)
    logger.info("CONFIGURATION")
    logger.info(f"  RAW_DATA_DIR={cfg.RAW_DATA_DIR}")
    logger.info(f"  PROCESSED_DATA_DIR={cfg.PROCESSED_DATA_DIR}")
    logger.info(f"  TRAIN_RATIO={cfg.TRAIN_RATIO} | VAL_RATIO={cfg.VAL_RATIO} | TEST_RATIO={cfg.TEST_RATIO}")

    raw_dir = Path(cfg.RAW_DATA_DIR)
    processed_dir = Path(cfg.PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download (ha kell)
    download_raw_data_if_needed(cfg)

    if not raw_dir.exists():
        logger.error(f"DATA: RAW_DATA_DIR not found: {raw_dir}")
        return

    # 2) Consensus neptunok (consensus/<NEPTUN>.json)
    consensus_root = raw_dir / "consensus"
    consensus_json_files = []

    if consensus_root.exists() and consensus_root.is_dir():
        consensus_json_files = sorted(consensus_root.glob("*.json"))

    # 3) Minden JSON összeszedése a szabályokkal
    #    - BP17IB kizárása
    #    - consensus fájlok mindig jönnek
    #    - rootban: ha top-level mappa neve benne van consensus_neptuns -> kihagyjuk a root mappát (dupe elkerülés)
    all_json = sorted(raw_dir.rglob("*.json"))
    if not all_json:
        logger.error("DATA: no JSON files found under RAW_DATA_DIR")
        return

    selected_json = []
    skipped_bp17ib = 0
    skipped_consensus_dupe_root = 0

    for jf in all_json:
        rel_parts = jf.relative_to(raw_dir).parts

        # BP17IB skip (case-sensitive path elem)
        if any(p == "BP17IB" for p in rel_parts):
            skipped_bp17ib += 1
            continue

        # consensus alatti json mindig mehet
        if rel_parts and rel_parts[0].lower() == "consensus":
            selected_json.append(jf)
            continue

        selected_json.append(jf)

    if not selected_json:
        logger.error("DATA: after filtering, no JSON files remain")
        return

    # 4) Beolvasás (csak összegző log!)
    all_samples = []
    bad_files = 0
    n_consensus_samples = 0
    n_other_samples = 0

    for jf in selected_json:
        rel_parts = jf.relative_to(raw_dir).parts
        is_consensus = rel_parts and rel_parts[0].lower() == "consensus"
        try:
            samples = load_labelstudio_json(jf)
        except Exception:
            bad_files += 1
            continue

        for text, label in samples:
            all_samples.append({"text": text, "label": label, "is_consensus": bool(is_consensus)})

        if is_consensus:
            n_consensus_samples += len(samples)
        else:
            n_other_samples += len(samples)

    if not all_samples:
        logger.error("DATA: no samples loaded from JSON files")
        return

    logger.info("LOADING RESULT")
    logger.info(f"  loaded_samples_total={len(all_samples)}")
    logger.info(f"  loaded_samples_consensus={n_consensus_samples}")
    logger.info(f"  loaded_samples_other={n_other_samples}")

    # 5) DataFrame + dedup
    df_all = pd.DataFrame(all_samples)
    label_dist_before = Counter(df_all["label"].tolist())

    df_all["text_norm"] = df_all["text"].apply(normalize_text_basic)
    unique_before = df_all["text_norm"].nunique()

    def aggregate_group(g: pd.DataFrame) -> pd.Series:
        # label = mode
        label = int(g["label"].mode().iloc[0])
        is_cons = bool(g["is_consensus"].any())
        # első szöveg elég (dedup célból a norm a lényeg)
        text = g["text"].iloc[0]
        return pd.Series({"text": text, "label": label, "is_consensus": is_cons})

    df_all_unique = (
        df_all.groupby("text_norm", as_index=False)
        .apply(aggregate_group)
        .reset_index(drop=True)
    )

    df_all = df_all_unique
    label_dist_after = Counter(df_all["label"].tolist())

    logger.info("DEDUPLICATION")
    logger.info(f"  rows_before={len(all_samples)}")
    logger.info(f"  unique_texts_before={unique_before}")
    logger.info(f"  rows_after={len(df_all)}")
    logger.info(f"  removed_duplicates={len(all_samples) - len(df_all)}")
    logger.info(f"  label_dist_before={label_dist_before}")
    logger.info(f"  label_dist_after={label_dist_after}")

    # 6) Split consensus vs other
    df_cons = df_all[df_all["is_consensus"]].reset_index(drop=True)
    df_other = df_all[~df_all["is_consensus"]].reset_index(drop=True)

    if len(df_cons) == 0:
        logger.error("DATA: no consensus samples after dedup -> cannot create consensus-based test")
        return

    # 7) Test set csak consensusból
    total_n = len(df_all)
    desired_n_test = int(round(cfg.TEST_RATIO * total_n))
    desired_n_test = max(1, desired_n_test)

    if desired_n_test >= len(df_cons):
        logger.warning("SPLIT: desired test size >= consensus size -> using ALL consensus as test")
        df_test = df_cons.copy()
        df_cons_rest = df_cons.iloc[0:0].copy()
    else:
        df_cons_rest, df_test = train_test_split(
            df_cons,
            test_size=desired_n_test,
            stratify=df_cons["label"],
            random_state=42,
        )

    # 8) Train/val pool
    df_trainval_pool = pd.concat([df_other, df_cons_rest], ignore_index=True)

    # Train/val split
    frac_val = cfg.VAL_RATIO / (cfg.TRAIN_RATIO + cfg.VAL_RATIO)
    df_train, df_val = train_test_split(
        df_trainval_pool,
        test_size=frac_val,
        stratify=df_trainval_pool["label"],
        random_state=42,
    )

    logger.info("SPLIT RESULT")
    logger.info(f"  train_n={len(df_train)} | val_n={len(df_val)} | test_n={len(df_test)}")
    logger.info(f"  train_label_dist={Counter(df_train['label'].tolist())}")
    logger.info(f"  val_label_dist={Counter(df_val['label'].tolist())}")
    logger.info(f"  test_label_dist={Counter(df_test['label'].tolist())}")

    # 9) Save
    train_out = processed_dir / "train.csv"
    val_out = processed_dir / "val.csv"
    test_out = processed_dir / "test.csv"

    df_train.to_csv(train_out, index=False)
    df_val.to_csv(val_out, index=False)
    df_test.to_csv(test_out, index=False)

    logger.info("OUTPUT")
    logger.info(f"  saved_train={train_out}")
    logger.info(f"  saved_val={val_out}")
    logger.info(f"  saved_test={test_out}")

    logger.info("==== Legal Text Decoder - PREPROCESS DONE ====")


if __name__ == "__main__":
    preprocess()
