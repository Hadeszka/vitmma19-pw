"""
Utility functions used across the Legal Text Decoder project.

Tartalmaz:
- logger setup (stdout)
- config loader
- egyszerű tokenizálás
- vocab építés / mentés / betöltés
- szöveg -> id-kódolás (padding/truncation)
- TinyLegalTextModel (maskos átlagolással)
"""

import logging
import sys
import re
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn


# ---------------------------
# Logger + config
# ---------------------------

def setup_logger(name=__name__):
    """
    Sets up a logger that outputs to the console (stdout).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config():
    import config
    return config


def log_kv(logger, title: str, d: dict):
    """
    Szép, tömör key=value blokk logolása.
    """
    logger.info(title)
    for k in sorted(d.keys()):
        logger.info(f"  {k} = {d[k]}")

# ---------------------------
# Tokenizálás + vocab
# ---------------------------

TOKEN_PATTERN = r"[^0-9a-záéíóöőúüű]+"

def simple_tokenize(text: str):
    """
    Egyszerű tokenizáló:
    - lower
    - nem alfanumerikus karakterek -> space
    - split
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(TOKEN_PATTERN, " ", text)
    return [t for t in text.split() if t]


def build_vocab(texts, max_vocab_size=20000, min_freq=1):
    """
    0: <pad>
    1: <unk>
    2..: leggyakoribb tokenek
    """
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    word2idx = {"<pad>": 0, "<unk>": 1}

    for w, f in counter.most_common():
        if f < min_freq:
            continue
        if len(word2idx) >= max_vocab_size:
            break
        word2idx[w] = len(word2idx)

    return word2idx


def save_vocab(word2idx: dict, vocab_path: Path):
    """
    Vocab mentése: soronként 1 token, az index = sorindex.
    (Ez kompatibilis az evaluation/inference load_vocab-val.)
    """
    vocab_path = Path(vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        for w, i in sorted(word2idx.items(), key=lambda x: x[1]):
            f.write(f"{w}\n")


def load_vocab(vocab_path: Path) -> dict:
    """
    Vocab betöltése: soronként 1 token.
    index = sorindex.
    """
    vocab_path = Path(vocab_path)
    stoi = {}
    with vocab_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            tok = line.rstrip("\n")
            if tok:
                stoi[tok] = idx
    return stoi


def encode_text(text: str, word2idx: dict, max_len: int):
    """
    Szöveg -> token id lista max_len hosszra (trunc/pad)
    """
    tokens = simple_tokenize(text)
    unk = word2idx.get("<unk>", 1)
    pad = word2idx.get("<pad>", 0)

    ids = [word2idx.get(t, unk) for t in tokens]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
    return ids


# ---------------------------
# Modell
# ---------------------------

class TinyLegalTextModel(nn.Module):
    """
    MLP baseline deep:
    Embedding -> masked mean -> ReLU -> Dropout -> Linear
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=32, num_classes=5, pad_idx=0, dropout=0.2):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)              # (B, L, D)
        mask = (input_ids != self.pad_idx).unsqueeze(-1)  # (B, L, 1)
        emb = emb * mask
        lengths = mask.sum(dim=1).clamp(min=1)       # (B, 1)
        pooled = emb.sum(dim=1) / lengths            # (B, D)
        h = self.relu(self.fc1(pooled))
        h = self.dropout(h)
        return self.fc2(h)


class TinyLegalTextCNN(nn.Module):
    """
    TextCNN:
    Embedding -> Conv1d(k in [3,4,5]) -> ReLU -> GlobalMaxPool -> concat -> Dropout -> Linear
    """
    def __init__(
        self,
        vocab_size,
        embed_dim=64,
        num_filters=64,
        kernel_sizes=(3, 4, 5),
        num_classes=5,
        pad_idx=0,
        dropout=0.2,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids):
        # input_ids: (B, L)
        x = self.embedding(input_ids)        # (B, L, D)
        x = x.transpose(1, 2)                # (B, D, L)  Conv1d expects channels-first

        pooled_list = []
        for conv in self.convs:
            h = self.relu(conv(x))           # (B, F, L-k+1)
            h = torch.max(h, dim=2).values   # global max pool -> (B, F)
            pooled_list.append(h)

        feat = torch.cat(pooled_list, dim=1) # (B, F * len(kernels))
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, outputs, mask):
        # outputs: (B, L, H)
        # mask:    (B, L)
        scores = self.attn(outputs).squeeze(-1)   # (B, L)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)        # (B, L)
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        return context, weights


class TinyLegalTextBiLSTMAttn(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        lstm_hidden_dim,
        num_classes,
        pad_idx=0,
        dropout=0.3,
        bidirectional=True,
        num_layers=1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            embed_dim,
            lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        self.attention = Attention(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, input_ids):
        # input_ids: (B, L)
        mask = (input_ids != 0).int()              # padding mask

        emb = self.embedding(input_ids)            # (B, L, D)
        outputs, _ = self.lstm(emb)                # (B, L, H)

        context, attn_weights = self.attention(outputs, mask)
        context = self.dropout(context)
        logits = self.fc(context)

        return logits
