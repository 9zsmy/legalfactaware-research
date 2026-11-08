#!/usr/bin/env python3
"""Time frozen TermDiffuSum decoding on the CAIL development split only."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--adapted-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bert-config-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    sys.path.insert(0, str(args.runtime_dir))
    import run_termdiffusum_cail_full as term

    # The recovered TermDiffuSum bytecode hard-codes ``bert-base-uncased``
    # when it builds the transformer skeleton.  Redirect that config lookup
    # to the archived local model so timing does not depend on the network.
    original_from_pretrained = AutoConfig.from_pretrained

    def local_from_pretrained(name_or_path, *values, **options):
        if str(name_or_path) == "bert-base-uncased":
            name_or_path = str(args.bert_config_dir)
        return original_from_pretrained(name_or_path, *values, **options)

    AutoConfig.from_pretrained = staticmethod(local_from_pretrained)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    config = term.model_and_diffusion_defaults()
    config.update(
        {
            "model_arch": "transformer_sum",
            "diffusion_steps": 500,
            "timestep_respacing": "",
            "noise_schedule": "legal_noise",
            "in_channel": 128,
            "out_channel": 128,
            "num_channels": 128,
            "predict_xstart": True,
            "training_mode": "e2e",
            "roc_train": "legal",
            "config_name": str(args.bert_config_dir),
        }
    )
    model, diffusion = term.create_model_and_diffusion(**config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = term.decode_split(
        "dev",
        model,
        diffusion,
        device,
        args.adapted_data_dir,
        args.output_dir,
        args.batch_size,
        8,
        450,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
