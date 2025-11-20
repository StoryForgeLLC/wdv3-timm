#!/usr/bin/env python3
"""
CCS WDv3 + timm tagger wrapper.

Features:
- Importable API: WDv3Tagger, WDv3Config
- Single-process GPU tagging with batching
- Backwards-compatible CLI:
    python ccs_wdv3_timm.py swinv2 /path/to/image.png
    -> prints one JSON dict to stdout

Also supports:
    python ccs_wdv3_timm.py --model swinv2 img1.png img2.png ...
    -> prints one JSON per line (JSONL)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import timm
from timm.data import create_transform, resolve_data_config
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
# HfHubHTTPError moved around between versions; be defensive.
try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    # Fallback: define a dummy so our `except HfHubHTTPError` blocks still work.
    class HfHubHTTPError(Exception):
        """Fallback when huggingface_hub doesn't expose HfHubHTTPError."""
        pass

# ---------------------------------------------------------------------------
# Model mapping (same keys you’ve been using: swinv2 / convnext / vit)
# ---------------------------------------------------------------------------

MODEL_REPO_MAP: Dict[str, str] = {
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "vit": "SmilingWolf/wd-vit-tagger-v3",
}


@dataclass
class LabelData:
    names: List[str]
    rating: List[int]
    general: List[int]
    character: List[int]


@dataclass
class WDv3Config:
    model_key: str               # "swinv2", "convnext", "vit"
    repo_id: str                 # HF repo id
    gen_threshold: float = 0.35  # default general tag threshold
    char_threshold: float = 0.75 # default character tag threshold
    revision: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure we end up with a clean RGB image (handles L, RGBA, etc.)."""
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")

    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    """Pad image to a square canvas with white background (centered)."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    """
    Download & parse selected_tags.csv from the WDv3 repo.

    The CSV has at least:
        name: tag name
        category: class group (rating/general/character)
    """
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
            token=token,
        )
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv not found in {repo_id}") from e

    csv_path = Path(csv_path).resolve()
    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])

    # These category ids follow WDv3’s convention:
    # - rating    -> category == 9
    # - general   -> category == 0
    # - character -> category == 4
    names = df["name"].tolist()
    categories = df["category"].to_numpy()

    rating_idx = list(np.where(categories == 9)[0])
    general_idx = list(np.where(categories == 0)[0])
    character_idx = list(np.where(categories == 4)[0])

    return LabelData(
        names=names,
        rating=rating_idx,
        general=general_idx,
        character=character_idx,
    )


def decode_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
):
    """
    Turn a probability vector into:
      caption, tags_display, ratings_dict, char_tags_dict, gen_tags_dict
    """
    arr = probs.detach().cpu().numpy()
    names = labels.names

    # Ratings: always keep full dict, sorted high->low
    rating_scores = {
        names[i]: float(arr[i]) for i in labels.rating
    }
    rating_sorted = dict(
        sorted(rating_scores.items(), key=lambda kv: kv[1], reverse=True)
    )

    # General tags above threshold
    general_pairs = [
        (names[i], float(arr[i]))
        for i in labels.general
        if float(arr[i]) > gen_threshold
    ]
    general_sorted = dict(
        sorted(general_pairs, key=lambda kv: kv[1], reverse=True)
    )

    # Character tags above threshold
    char_pairs = [
        (names[i], float(arr[i]))
        for i in labels.character
        if float(arr[i]) > char_threshold
    ]
    char_sorted = dict(
        sorted(char_pairs, key=lambda kv: kv[1], reverse=True)
    )

    # Caption = general + character + best rating tag
    tokens: List[str] = list(general_sorted.keys()) + list(char_sorted.keys())
    if rating_sorted:
        top_rating = next(iter(rating_sorted.keys()))
        tokens.append("rating_" + top_rating)

    caption = ", ".join(tokens)
    tags_display = caption.replace("_", " ")

    return caption, tags_display, rating_sorted, char_sorted, general_sorted


# ---------------------------------------------------------------------------
# Core tagger class
# ---------------------------------------------------------------------------

class WDv3Tagger:
    """
    Wraps a WDv3 timm classifier (swinv2/convnext/vit) for batch tagging.
    Keeps the model on GPU and runs batched inference.
    """

    def __init__(
        self,
        config: WDv3Config,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load label metadata
        self.labels = load_labels_hf(
            repo_id=config.repo_id,
            revision=config.revision,
            token=hf_token,
        )

        # Build model from HF hub via timm
        model_name = "hf-hub:" + config.repo_id
        model: nn.Module = timm.create_model(model_name, pretrained=False)

        try:
            # Newer timm supports revision arg; older versions ignore it
            state_dict = timm.models.load_state_dict_from_hf(
                config.repo_id,
                revision=config.revision,
            )
        except TypeError:
            state_dict = timm.models.load_state_dict_from_hf(config.repo_id)

        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        self.model = model

        # Build data transform from model’s pretrained config
        data_cfg = resolve_data_config(model.pretrained_cfg, model=model)
        self.transform = create_transform(**data_cfg)

    # --- Internal helpers ---------------------------------------------------

    def _prepare_tensor(self, image_path: Path) -> Optional[Tensor]:
        try:
            img = Image.open(image_path)
        except (FileNotFoundError, UnidentifiedImageError):
            return None

        img = pil_ensure_rgb(img)
        img = pil_pad_square(img)
        t = self.transform(img)
        # WDv3 models expect BGR order
        t = t[[2, 1, 0], ...]
        return t

    # --- Public API ---------------------------------------------------------

    def tag_image(self, image_path: Path | str) -> Dict[str, Any]:
        """Tag a single image path and return one JSON-serializable dict."""
        results = self.tag_paths([image_path], batch_size=1)
        return results[0]

    def tag_paths(
        self,
        image_paths: Sequence[Path | str],
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Tag many image paths in batches, keeping the model on GPU.

        Returns a list of dicts:
          {
            backend, model, repo_id,
            image_path, caption, tags_display,
            ratings, character_tags, general_tags,
            gen_threshold, char_threshold
          }
        """
        paths = [Path(p) for p in image_paths]
        out: List[Dict[str, Any]] = []

        batch_tensors: List[Tensor] = []
        batch_meta: List[Path] = []

        def flush_batch():
            nonlocal batch_tensors, batch_meta, out
            if not batch_tensors:
                return

            batch = torch.stack(batch_tensors, dim=0).to(self.device)

            with torch.inference_mode():
                logits: Tensor = self.model(batch)
                probs: Tensor = torch.sigmoid(logits)

            probs = probs.cpu()

            for prob_vec, path in zip(probs, batch_meta):
                (
                    caption,
                    tags_display,
                    ratings,
                    char_tags,
                    gen_tags,
                ) = decode_tags(
                    prob_vec,
                    self.labels,
                    self.config.gen_threshold,
                    self.config.char_threshold,
                )

                out.append(
                    {
                        "backend": "wdv3-timm",
                        "model": self.config.model_key,
                        "repo_id": self.config.repo_id,
                        "image_path": str(path),
                        "caption": caption,
                        "tags_display": tags_display,
                        "ratings": ratings,
                        "character_tags": char_tags,
                        "general_tags": gen_tags,
                        "gen_threshold": self.config.gen_threshold,
                        "char_threshold": self.config.char_threshold,
                    }
                )

            batch_tensors = []
            batch_meta = []

        for p in paths:
            t = self._prepare_tensor(p)
            if t is None:
                # Emit an error record instead of crashing
                out.append(
                    {
                        "backend": "wdv3-timm",
                        "model": self.config.model_key,
                        "repo_id": self.config.repo_id,
                        "image_path": str(p),
                        "error": "failed_to_load_image",
                    }
                )
                continue

            batch_tensors.append(t)
            batch_meta.append(p)

            if len(batch_tensors) >= batch_size:
                flush_batch()

        # Flush remainder
        flush_batch()
        return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _legacy_cli(argv: List[str]) -> None:
    """
    Support the old call style:

        python ccs_wdv3_timm.py swinv2 /path/to/image.png

    Prints ONE JSON dict to stdout.
    """
    model_key = argv[1]
    image_path = argv[2]

    if model_key not in MODEL_REPO_MAP:
        raise SystemExit(f"Unknown model '{model_key}'. Use one of: {', '.join(MODEL_REPO_MAP.keys())}")

    cfg = WDv3Config(
        model_key=model_key,
        repo_id=MODEL_REPO_MAP[model_key],
        gen_threshold=0.35,
        char_threshold=0.75,
    )
    tagger = WDv3Tagger(cfg)
    result = tagger.tag_image(image_path)
    print(json.dumps(result))


def _argparse_cli(argv: List[str]) -> None:
    """
    Newer CLI:

        python ccs_wdv3_timm.py --model swinv2 img1.png img2.png ...

    Prints one JSON per line (JSONL).
    """
    import argparse

    parser = argparse.ArgumentParser(description="CCS WDv3 timm tagger")
    parser.add_argument(
        "--model",
        "-m",
        choices=list(MODEL_REPO_MAP.keys()),
        default="swinv2",
        help="Tagger backbone to use",
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image files to tag",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="Batch size for GPU inference",
    )
    parser.add_argument(
        "--gen-threshold",
        type=float,
        default=0.35,
        help="General tag probability threshold",
    )
    parser.add_argument(
        "--char-threshold",
        type=float,
        default=0.75,
        help="Character tag probability threshold",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="HF revision / commit id (optional)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HF token if needed (optional)",
    )

    args = parser.parse_args(argv[1:])

    cfg = WDv3Config(
        model_key=args.model,
        repo_id=MODEL_REPO_MAP[args.model],
        gen_threshold=args.gen_threshold,
        char_threshold=args.char_threshold,
        revision=args.revision,
    )
    tagger = WDv3Tagger(cfg, device=args.device, hf_token=args.hf_token)

    results = tagger.tag_paths(args.images, batch_size=args.batch_size)
    for r in results:
        print(json.dumps(r))


def main() -> None:
    # If called as: python ccs_wdv3_timm.py swinv2 image.png
    if len(sys.argv) == 3 and not sys.argv[1].startswith("-"):
        _legacy_cli(sys.argv)
    else:
        _argparse_cli(sys.argv)


if __name__ == "__main__":
    main()
