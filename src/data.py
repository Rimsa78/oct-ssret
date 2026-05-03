"""OCT data layer: Kermany 2017 (4-class CNV/DME/DRUSEN/NORMAL),
OCTID (cross-cohort), and ekacare (paired fundus+OCT+layer annotations).

Kermany is the primary training+in-domain test source. OCTID and ekacare
are zero-shot cross-cohort test sets.
"""
from __future__ import annotations
import io
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import cv2

# Kermany classes (canonical order from the HF mirror's ClassLabel)
KERMANY_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]


def _to_numpy_image(img) -> np.ndarray:
    """Convert PIL Image / bytes / np.ndarray -> uint8 (H, W, 3) BGR or grayscale 3-ch."""
    if isinstance(img, np.ndarray):
        arr = img
    elif isinstance(img, (bytes, bytearray)):
        arr = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    elif hasattr(img, "convert"):
        arr = np.array(img.convert("L"))  # OCT is grayscale; keep as L
    else:
        raise TypeError(type(img))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def _resize_pad_to_square(img: np.ndarray, image_size: int) -> np.ndarray:
    """Letterbox-resize an arbitrary-aspect OCT B-scan to a square at image_size.
    Pads the shorter axis with the image median to keep aspect ratio sane.
    """
    h, w = img.shape[:2]
    scale = image_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_v = image_size - nh
    pad_h = image_size - nw
    pad_top = pad_v // 2
    pad_bot = pad_v - pad_top
    pad_left = pad_h // 2
    pad_right = pad_h - pad_left
    pad_color = int(np.median(resized))
    out = cv2.copyMakeBorder(resized, pad_top, pad_bot, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=(pad_color,) * 3)
    return out


def _light_clahe(img: np.ndarray) -> np.ndarray:
    """Apply mild CLAHE on the green channel (or grayscale) for contrast normalisation."""
    if img.ndim == 3:
        g = img[..., 1]
    else:
        g = img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(g.astype(np.uint8))
    if img.ndim == 3:
        img = img.copy()
        img[..., 0] = out
        img[..., 1] = out
        img[..., 2] = out
    else:
        img = np.stack([out, out, out], axis=-1)
    return img


# -----------------------------------------------------------------------------
# Datasets
# -----------------------------------------------------------------------------

class KermanyOCTDataset(Dataset):
    """Wraps `zacharielegault/Kermany2017-OCT`. 4-class CNV/DME/DRUSEN/NORMAL."""
    def __init__(self, rows, image_size: int = 256, is_train: bool = False):
        self.rows = rows
        self.image_size = image_size
        self.is_train = is_train
        # build train-time light augmentation lazily
        if is_train:
            import albumentations as A
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(translate_percent=0.04, scale=(0.92, 1.08), rotate=(-8, 8),
                         interpolation=cv2.INTER_LINEAR, p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = _to_numpy_image(row["image"])
        img = _resize_pad_to_square(img, self.image_size)
        img = _light_clahe(img)
        if self.aug is not None:
            img = self.aug(image=img)["image"]
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {"image": img_t, "label": int(row["label"])}


class OCTIDDataset(Dataset):
    """Wraps `ai4ophth/OCTID_dataset` for zero-shot cross-cohort testing.
    OCTID has multiple disease classes encoded in a metadata dict; we map them
    to the Kermany 4-class scheme where possible:
        Normal               -> 3 (NORMAL)
        AMD                  -> 2 (DRUSEN)   (AMD ~ drusen-bearing)
        DR (diabetic retin.) -> 1 (DME)      (DME-like presentations)
        MH (macular hole)    -> 0 (CNV)      (closest analog; flagged as approx)
        CSR                  -> -1           (sentinel, dropped)
    The mapping is approximate and is reported transparently in the paper.
    """
    # The OCTID HF mirror has 5 disease classes encoded in the 'sparse text'
    # field as natural-language disease names. We map each to the closest
    # Kermany 4-class label; CSR has no Kermany analog and is dropped.
    # Kermany classes: 0=CNV, 1=DME, 2=DRUSEN, 3=NORMAL.
    OCTID_KEYWORDS = [
        # (sparse-text substring, Kermany label, disclosure)
        ("normal",                        3),
        ("age-related macular",           2),  # AMD (often drusen-bearing) -> DRUSEN
        ("age-related",                   2),
        ("diabetic retinopathy",          1),  # DR -> DME (most clinically aligned Kermany class)
        ("diabetic",                      1),
        ("macular hole",                  0),  # MH (closest analog to CNV: macular pathology)
        ("central serous",               -1),  # CSR -> sentinel (drop)
        ("central serous retinopathy",   -1),
    ]

    def __init__(self, rows, image_size: int = 256):
        self.rows = []
        for r in rows:
            label = self._extract_label(r)
            if label is not None and label >= 0:
                self.rows.append((r, label))
        self.image_size = image_size

    def _extract_label(self, r):
        st = (r.get("sparse text") or "").lower()
        for kw, lab in self.OCTID_KEYWORDS:
            if kw in st:
                return lab
        return None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row, label = self.rows[idx]
        img_field = row.get("image") or row.get("fundus images")
        img = _to_numpy_image(img_field)
        img = _resize_pad_to_square(img, self.image_size)
        img = _light_clahe(img)
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {"image": img_t, "label": int(label)}


class EkacareOCTDataset(Dataset):
    """Wraps `ekacare/OCT_And_Fundus_Glaucoma_Dataset`. Small (50), paired with
    fundus + ILM/RPE annotations; used here for qualitative figures only.

    We aggregate the Glaucoma label across the 4 annotators (Opt_1..Opt_4) by
    majority vote: 'yes' -> 1, 'no' -> 0, 'suspect' -> dropped.
    """
    def __init__(self, rows, image_size: int = 256):
        self.rows = []
        for r in rows:
            label = self._aggregate_label(r)
            if label is not None:
                self.rows.append((r, label))
        self.image_size = image_size

    def _aggregate_label(self, r):
        votes = []
        for opt in ["Opt_1", "Opt_2", "Opt_3", "Opt_4"]:
            d = r.get(opt) or {}
            g = (d.get("Glaucoma") or "").lower()
            if g == "yes":   votes.append(1)
            elif g == "no":  votes.append(0)
        if not votes:
            return None
        return int(round(sum(votes) / len(votes)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row, label = self.rows[idx]
        img = _to_numpy_image(row["B-scan"])
        img = _resize_pad_to_square(img, self.image_size)
        img = _light_clahe(img)
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {"image": img_t, "label": int(label),
                "ilm": np.array(row["B-scan_ILM"].convert("L")),
                "rpe": np.array(row["B-scan_RPE"].convert("L"))}


# -----------------------------------------------------------------------------
# DataLoader factories
# -----------------------------------------------------------------------------

def get_kermany_loaders(batch_size: int = 32, image_size: int = 256,
                        num_workers: int = 4, val_frac: float = 0.10,
                        seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader) on Kermany. We carve a
    val split from the 108k train set (10% by default), and use the held-out
    1k test split as is.
    """
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    ds = load_dataset("zacharielegault/Kermany2017-OCT")
    train_full = ds["train"]
    test_ds = ds["test"]

    n = len(train_full)
    idx = np.arange(n)
    labels = np.array([int(train_full[int(i)]["label"]) for i in idx])
    tr_idx, va_idx = train_test_split(idx, test_size=val_frac, stratify=labels,
                                       random_state=seed)
    train_rows = train_full.select(tr_idx.tolist())
    val_rows = train_full.select(va_idx.tolist())

    train_ds = KermanyOCTDataset(train_rows, image_size=image_size, is_train=True)
    val_ds   = KermanyOCTDataset(val_rows,   image_size=image_size, is_train=False)
    test_ds_ = KermanyOCTDataset(test_ds,    image_size=image_size, is_train=False)

    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=False)
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, drop_last=False)
    te = DataLoader(test_ds_, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, drop_last=False)
    return tr, va, te


class OCTDLDataset(Dataset):
    """Wraps `ArmisticeAI/OCTDL2024` for zero-shot cross-cohort testing.
    OCTDL has 5 disease classes; we map to the Kermany 4-class scheme.
    Mapping (mirrors the OCTID convention):
        NORMAL  -> 3 (NORMAL)
        AMD     -> 2 (DRUSEN, AMD ~ drusen-bearing)
        DR      -> 1 (DME, closest analog)
        MH      -> 0 (CNV, closest macular pathology analog)
        RVO     -> -1 (drop, no Kermany analog)
    """
    OCTDL_TO_KERMANY = {
        # OCTDL ClassLabel order: ['MH', 'DR', 'AMD', 'NORMAL', 'RVO']
        0: 0,   # MH -> CNV
        1: 1,   # DR -> DME
        2: 2,   # AMD -> DRUSEN
        3: 3,   # NORMAL -> NORMAL
        4: -1,  # RVO -> drop
    }

    def __init__(self, rows, image_size: int = 256):
        self.rows = []
        for r in rows:
            mapped = self.OCTDL_TO_KERMANY.get(int(r["label"]))
            if mapped is not None and mapped >= 0:
                self.rows.append((r, mapped))
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row, label = self.rows[idx]
        img = _to_numpy_image(row["image"])
        img = _resize_pad_to_square(img, self.image_size)
        img = _light_clahe(img)
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return {"image": img_t, "label": int(label)}


def get_octdl_loader(batch_size: int = 16, image_size: int = 256, num_workers: int = 2):
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset("ArmisticeAI/OCTDL2024")
    # use train+val+test combined as a zero-shot transfer cohort (we never train on OCTDL)
    combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    octdl_ds = OCTDLDataset(combined, image_size=image_size)
    return DataLoader(octdl_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_octid_loader(batch_size: int = 32, image_size: int = 256, num_workers: int = 2):
    from datasets import load_dataset
    ds = load_dataset("ai4ophth/OCTID_dataset", split="train")
    octid_ds = OCTIDDataset(ds, image_size=image_size)
    return DataLoader(octid_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def get_ekacare_loader(batch_size: int = 4, image_size: int = 256, num_workers: int = 2):
    from datasets import load_dataset
    ds = load_dataset("ekacare/OCT_And_Fundus_Glaucoma_Dataset", split="train")
    eka_ds = EkacareOCTDataset(ds, image_size=image_size)
    return DataLoader(eka_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)
