#!/usr/bin/env python3
"""
WD Tagger v3 + timm, JSON output version for Comic Creator Studio.

Usage:
  python ccs_wdv3_timm.py --model swinv2 path/to/image.png > tags.json

Models:
  swinv2   -> SmilingWolf/wd-swinv2-tagger-v3
  vit      -> SmilingWolf/wd-vit-tagger-v3
  convnext -> SmilingWolf/wd-convnext-tagger-v3
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import timm
from timm.data import resolve_model_data_config, create_transform
from safetensors.torch import load_file as load_safetensors


MODEL_REPOS: Dict[str, str] = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


@dataclass
class LabelData:
    names: List[str]
    rating_indices: List[int]
    general_indices: List[int]
    character_indices: List[int]


def load_labels_hf(repo_id: str) -> LabelData:
    """
    Download and parse selected_tags.csv from the given HF repo.
    Categories (per WD v3):
      0 -> general
      4 -> character
      9 -> rating
    """
    csv_path: str = hf_hub_download(repo_id, "selected_tags.csv")

    names: List[str] = []
    rating_indices: List[int] = []
    general_indices: List[int] = []
    character_indices: List[int] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            name: str = row["name"]
            category: int = int(row["category"])

            names.append(name)

            if category == 9:
                rating_indices.append(idx)
            elif category == 0:
                general_indices.append(idx)
            elif category == 4:
                character_indices.append(idx)

    return LabelData(
        names=names,
        rating_indices=rating_indices,
        general_indices=general_indices,
        character_indices=character_indices,
    )


def load_model_and_transform(
    model_key: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, object, str]:
    """
    Load WD v3 timm model + data transform from HF config + weights.
    """
    if model_key not in MODEL_REPOS:
        raise ValueError(f"Unknown model key '{model_key}'. Use one of: {list(MODEL_REPOS.keys())}.")

    repo_id: str = MODEL_REPOS[model_key]

    # Read config.json from HF – contains timm architecture + num_classes
    config_path: str = hf_hub_download(repo_id, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    architecture: str = config["architecture"]
    num_classes: int = int(config["num_classes"])

    # Create timm model
    model: torch.nn.Module = timm.create_model(
        architecture,
        pretrained=False,
        num_classes=num_classes,
    )

    # Load safetensors weights
    weights_path: str = hf_hub_download(repo_id, "model.safetensors")
    state_dict = load_safetensors(weights_path)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    model.to(device)

    # Build eval transform from timm data config
    data_cfg = resolve_model_data_config(model)
    transform = create_transform(
        **data_cfg,
        is_training=False,
    )

    return model, transform, repo_id


def run_inference(
    image_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> np.ndarray:
    """
    Load image, preprocess, run model, return probabilities as numpy array.
    """
    image = Image.open(image_path).convert("RGB")
    tensor: torch.Tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits: torch.Tensor = model(tensor)
        # WD tagger is multi-label; apply sigmoid over logits
        probs_tensor: torch.Tensor = torch.sigmoid(logits)[0].cpu()

    probs: np.ndarray = probs_tensor.numpy()
    return probs


def build_tags_json(
    probs: np.ndarray,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
    model_key: str,
    repo_id: str,
    image_path: str,
) -> Dict[str, object]:
    """
    Convert raw probabilities into a JSON-ready dict with:
      - caption (wd-style tags)
      - tags_display (underscores to spaces, parens escaped)
      - ratings, character_tags, general_tags dicts
    """

    # All (name, score) pairs
    prob_list: List[float] = probs.tolist()
    name_score_pairs: List[Tuple[str, float]] = list(zip(labels.names, prob_list))

    # Ratings (4 outputs)
    ratings: Dict[str, float] = {}
    for idx in labels.rating_indices:
        tag_name, score = name_score_pairs[idx]
        ratings[tag_name] = float(score)

    # General tags (thresholded + sorted desc)
    general_tags: Dict[str, float] = {}
    for idx in labels.general_indices:
        tag_name, score = name_score_pairs[idx]
        if score >= gen_threshold:
            general_tags[tag_name] = float(score)

    # Sort by descending score
    general_tags = dict(sorted(general_tags.items(), key=lambda item: item[1], reverse=True))

    # Character tags (thresholded + sorted desc)
    character_tags: Dict[str, float] = {}
    for idx in labels.character_indices:
        tag_name, score = name_score_pairs[idx]
        if score >= char_threshold:
            character_tags[tag_name] = float(score)

    character_tags = dict(sorted(character_tags.items(), key=lambda item: item[1], reverse=True))

    # Build caption string (wd-style names)
    combined_names: List[str] = list(general_tags.keys()) + list(character_tags.keys())
    caption: str = ", ".join(combined_names)

    # Build display taglist (underscores to spaces, parens escaped)
    taglist: str = caption.replace("_", " ")
    taglist = taglist.replace("(", r"\(").replace(")", r"\)")

    result: Dict[str, object] = {
        "backend": "wdv3-timm",
        "model": model_key,
        "repo_id": repo_id,
        "image_path": os.path.abspath(image_path),
        "caption": caption,
        "tags_display": taglist,
        "ratings": ratings,
        "character_tags": character_tags,
        "general_tags": general_tags,
        "gen_threshold": gen_threshold,
        "char_threshold": char_threshold,
    }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WD Tagger v3 via timm and emit JSON.",
    )
    parser.add_argument(
        "model",
        choices=list(MODEL_REPOS.keys()),
        help="Which WD v3 model backbone to use.",
    )
    parser.add_argument(
        "image",
        help="Path to input image.",
    )
    parser.add_argument(
        "--gen-threshold",
        type=float,
        default=0.35,
        help="Probability threshold for general tags (default: 0.35).",
    )
    parser.add_argument(
        "--char-threshold",
        type=float,
        default=0.75,
        help="Probability threshold for character tags (default: 0.75).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path: str = args.image
    model_key: str = args.model
    gen_threshold: float = args.gen_threshold
    char_threshold: float = args.char_threshold
    pretty: bool = args.pretty

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    device: torch.device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model + labels
    model, transform, repo_id = load_model_and_transform(model_key, device)
    labels: LabelData = load_labels_hf(repo_id)

    # Run inference
    probs: np.ndarray = run_inference(image_path, model, transform, device)

    # Convert to structured JSON dict
    result: Dict[str, object] = build_tags_json(
        probs=probs,
        labels=labels,
        gen_threshold=gen_threshold,
        char_threshold=char_threshold,
        model_key=model_key,
        repo_id=repo_id,
        image_path=image_path,
    )

    if pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
